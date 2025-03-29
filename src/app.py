"""
Main entry point for the Video Game Sales Dashboard application with performance optimization
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import logging
import os
import asyncio
from pathlib import Path
from logging.handlers import RotatingFileHandler
import time
import gc
import numpy as np
import traceback
import platform
import psutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
import threading
from typing import Dict, Optional, Any, List, Union, Tuple, Callable

# Import app components
from src.layouts.main_layout import create_layout
from src.data.data_loader import load_data, load_data_async
from src.utils.cache import DataFrameCache
from src.callbacks.register_callbacks import register_all_callbacks

# Configure logging with rotation
def setup_app_logging():
    """
    Set up application logging with rotation and performance tracking
    
    Returns:
        Logger instance for the application
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure handlers
    file_handler = RotatingFileHandler(
        log_dir / "dashboard.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    console_handler = logging.StreamHandler()
    
    # Set formats
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

# Set up logging
logger = setup_app_logging()

# Available themes with descriptions
THEMES = {
    "Light": {
        "theme": dbc.themes.BOOTSTRAP,
        "description": "Standard Bootstrap theme with clean, professional look"
    },
    "Dark": {
        "theme": dbc.themes.DARKLY,
        "description": "Dark theme for reduced eye strain in low-light environments"
    },
    "Slate": {
        "theme": dbc.themes.SLATE,
        "description": "Dark blue-gray theme with subtle color accents"
    },
    "Superhero": {
        "theme": dbc.themes.SUPERHERO,
        "description": "Bold blue theme with strong contrasts"
    }
}

# Plotly config options
DEFAULT_PLOTLY_CONFIG = {
    "use_custom_templates": True,
    "simple_charts": False
}

class PerformanceMonitor:
    """Monitor application performance and resource usage"""
    def __init__(self):
        """Initialize the performance monitor"""
        self.start_time = time.time()
        self.metrics: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.monitoring_active = True
        self._start_background_monitoring()
    
    def _start_background_monitoring(self) -> None:
        """Start background thread for continuous monitoring"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.optimize_memory()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(120)  # Wait longer after error
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Background performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the background monitoring"""
        self.monitoring_active = False
    
    def log_system_info(self) -> None:
        """
        Log system information for diagnostics
        """
        try:
            logger.info(f"System: {platform.system()} {platform.release()}")
            logger.info(f"Python: {platform.python_version()}")
            
            virtual_memory = psutil.virtual_memory()
            logger.info(f"Memory: Total={virtual_memory.total/(1024**3):.1f} GB, "
                       f"Available={virtual_memory.available/(1024**3):.1f} GB")
            
            cpu_count = psutil.cpu_count(logical=False) or 1
            logical_cpus = psutil.cpu_count() or 1
            logger.info(f"CPU: {cpu_count} cores, {logical_cpus} logical processors")
            
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Check disk space
            disk_usage = psutil.disk_usage(os.getcwd())
            logger.info(f"Disk: Total={disk_usage.total/(1024**3):.1f} GB, "
                      f"Free={disk_usage.free/(1024**3):.1f} GB")
            
            # GPU info if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu.name}, Memory: {gpu.memoryTotal} MB")
            except (ImportError, Exception):
                pass  # GPU utils not available or error occurred
            
        except Exception as e:
            logger.warning(f"Could not gather all system info: {str(e)}")
    
    def optimize_memory(self) -> None:
        """
        Run garbage collection and optimize memory usage
        """
        try:
            # Force garbage collection
            collected = gc.collect(generation=2)  # Full collection
            logger.debug(f"Garbage collection: collected {collected} objects")
            
            # Get memory info
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Update metrics
            with self.lock:
                self.metrics['memory_usage_mb'] = memory_mb
                self.metrics['garbage_collected'] = collected
            
            # Log memory pressure if high
            if memory_mb > 1000:  # If over 1GB
                logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                
                # More aggressive memory optimization
                if memory_mb > 2000:  # If over 2GB
                    logger.warning("Critical memory pressure - forcing additional cleanup")
                    gc.collect(generation=2)
                    
                    # Reduce thread pool size if needed
                    if len(self.executor._threads) > 4:
                        self.executor.shutdown(wait=False)
                        self.executor = ThreadPoolExecutor(max_workers=2)
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {str(e)}")
    
    def record_metric(self, name: str, value: float) -> None:
        """
        Record a performance metric
        
        Args:
            name: Name of the metric
            value: Value to record
        """
        with self.lock:
            self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all recorded metrics
        
        Returns:
            Dictionary of metrics
        """
        with self.lock:
            metrics_copy = self.metrics.copy()
            # Add uptime
            metrics_copy['uptime_seconds'] = self.get_uptime()
            return metrics_copy
    
    def get_uptime(self) -> float:
        """
        Get application uptime in seconds
        
        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time
    
    def format_metrics_for_display(self) -> str:
        """
        Format metrics for display in the UI
        
        Returns:
            Formatted metrics string
        """
        metrics = self.get_metrics()
        uptime_seconds = metrics.get('uptime_seconds', 0)
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        formatted = [
            f"Uptime: {int(hours)}h {int(minutes)}m {int(seconds)}s",
            f"Memory: {metrics.get('memory_usage_mb', 0):.1f} MB",
            f"CPU Usage: {psutil.cpu_percent(interval=0.1):.1f}%",
        ]
        
        # Add cache stats if available
        if 'cache_hit_rate' in metrics:
            formatted.append(f"Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1f}%")
        
        return "\n".join(formatted)

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

def create_app(memory_limit_mb: Optional[int] = None, cache_size: int = 20,
              disable_custom_templates: bool = False, simple_charts: bool = False) -> dash.Dash:
    """
    Create and configure the Dash application (synchronous wrapper)
    
    Args:
        memory_limit_mb: Memory limit in MB for cache
        cache_size: Size of the cache
        disable_custom_templates: Disable Plotly custom templates to fix compatibility issues
        simple_charts: Use simple chart configurations for better compatibility
        
    Returns:
        Configured Dash application
    """
    # Configure plotly
    plotly_config = DEFAULT_PLOTLY_CONFIG.copy()
    if disable_custom_templates:
        plotly_config["use_custom_templates"] = False
    if simple_charts:
        plotly_config["simple_charts"] = True
        
    logger.info(f"Starting app with Plotly config: {plotly_config}")
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    performance_monitor.log_system_info()
    
    # Log metrics and system information
    logger.info("Initializing application...")
    start_time = time.time()
    
    try:
        # Load data synchronously
        df, cache = load_data(cache_size=cache_size, memory_limit_mb=memory_limit_mb)
        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        
        # Create and configure the Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[THEMES["Light"]["theme"]],
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        
        # Set up the app layout
        app.layout = create_layout(df)
        
        # Register callbacks
        register_all_callbacks(app, df, cache, plotly_config)
        
        # Register teardown
        @app.server.teardown_appcontext
        def shutdown_cleanup(exception=None):
            """Clean up resources when app is shutting down"""
            try:
                performance_monitor.stop_monitoring()
                logger.info("Shutting down performance monitoring")
                
                # Force garbage collection
                gc.collect()
                
                logger.info(f"Application shutdown after {performance_monitor.get_uptime():.1f} seconds")
            except Exception as e:
                logger.error(f"Error during shutdown: {str(e)}")
        
        return app
    
    except Exception as e:
        logger.error(f"Error creating app: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def async_create_app(memory_limit_mb: Optional[int] = None, cache_size: int = 20, 
                          disable_custom_templates: bool = False, simple_charts: bool = False) -> dash.Dash:
    """
    Asynchronously create and configure the Dash application
    
    Args:
        memory_limit_mb: Memory limit in MB for cache
        cache_size: Size of the cache
        disable_custom_templates: Disable Plotly custom templates to fix compatibility issues
        simple_charts: Use simple chart configurations to improve compatibility
        
    Returns:
        Configured Dash application
    """
    # Configure plotly
    plotly_config = DEFAULT_PLOTLY_CONFIG.copy()
    if disable_custom_templates:
        plotly_config["use_custom_templates"] = False
    if simple_charts:
        plotly_config["simple_charts"] = True
        
    logger.info(f"Starting app with Plotly config: {plotly_config}")
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    performance_monitor.log_system_info()
    
    # Log metrics and system information
    logger.info("Initializing application...")
    start_time = time.time()
    
    try:
        # Load the data asynchronously
        df, cache = await load_data_async(cache_size=cache_size, memory_limit_mb=memory_limit_mb)
        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        
        # Create and configure the Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[THEMES["Light"]["theme"]],
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        
        # Set up the app layout
        app.layout = create_layout(df)
        
        # Register callbacks
        register_all_callbacks(app, df, cache, plotly_config)
        
        # Register teardown
        @app.server.teardown_appcontext
        def shutdown_cleanup(exception=None):
            """Clean up resources when app is shutting down"""
            try:
                performance_monitor.stop_monitoring()
                logger.info("Shutting down performance monitoring")
                
                # Force garbage collection
                gc.collect()
                
                logger.info(f"Application shutdown after {performance_monitor.get_uptime():.1f} seconds")
            except Exception as e:
                logger.error(f"Error during shutdown: {str(e)}")
        
        return app
    
    except Exception as e:
        logger.error(f"Error creating app: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Run the application for development/testing"""
    app = create_app()
    app.run_server(debug=True)

if __name__ == "__main__":
    main() 