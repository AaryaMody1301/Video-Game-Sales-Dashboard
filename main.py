#!/usr/bin/env python3
"""
Main entry point for the Video Game Sales Dashboard application.
Provides command-line interface for running the dashboard with various options.

This application provides interactive visualizations and analysis of video game sales
data using Dash and Plotly. It includes features for filtering, forecasting,
and exporting data in various formats.

Usage:
    python main.py [--debug] [--port PORT] [--host HOST] [--workers WORKERS]

Options:
    --debug     Run the application in debug mode (default: False)
    --port      Specify the port to run the application on (default: 8050)
    --host      Specify the host to run the application on (default: 127.0.0.1)
    --workers   Number of worker processes (default: 1)

Examples:
    python main.py
    python main.py --debug --port 8000 --host 0.0.0.0 --workers 4

Author: Video Game Sales Dashboard Team
License: MIT
"""
import argparse
import logging
import os
import sys
import time
import gc
import traceback
import platform
import psutil
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
from typing import Dict, Optional, List, Any, Union, Tuple, cast
from datetime import datetime
import multiprocessing
from pathlib import Path

# Import app components
from src.app import create_app, performance_monitor, async_create_app

# Windows-specific path handling
def get_project_root() -> Path:
    """Get the project root directory in a cross-platform way."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).resolve().parent

def get_log_dir() -> Path:
    """Get the log directory path in a cross-platform way."""
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir

# Configure logging with rotation
def setup_logging(log_dir: Optional[str] = None) -> None:
    """
    Configure logging with file rotation and console output.
    
    Args:
        log_dir: Optional path to log directory. If None, uses default path.
    """
    if log_dir is None:
        log_dir = get_log_dir()
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Import the handler directly to avoid attribute error
    from logging.handlers import RotatingFileHandler
    
    # Create handlers
    current_time = datetime.now()
    log_file = Path(log_dir) / f"dashboard_{current_time.strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set formats
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Game Sales Dashboard')
    
    # Server configuration
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    # Performance options
    parser.add_argument('--memory-limit', type=int, help='Memory limit in MB for cache', default=None)
    parser.add_argument('--cache-size', type=int, default=20, help='Maximum items in cache')
    
    # Logging options
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set logging level')
    
    # Chart compatibility options
    parser.add_argument('--disable-custom-templates', action='store_true',
                      help='Disable custom Plotly templates to fix compatibility issues')
    parser.add_argument('--simple-charts', action='store_true',
                      help='Use simplified chart configurations for better compatibility')
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If any argument is invalid
    """
    if args.port < 1 or args.port > 65535:
        raise ValueError("Port must be between 1 and 65535")
    if args.workers < 1:
        raise ValueError("Number of workers must be at least 1")
    
    # Check if workers exceed CPU count and adjust if needed
    cpu_count = multiprocessing.cpu_count()
    if args.workers > cpu_count:
        logging.warning(f"Number of workers ({args.workers}) exceeds CPU count ({cpu_count}). "
                      f"Using {cpu_count} workers instead.")
        args.workers = cpu_count
    
    # Validate memory limit
    if args.memory_limit is not None and args.memory_limit < 50:
        logging.warning(f"Memory limit {args.memory_limit}MB is very low, setting to 50MB minimum")
        args.memory_limit = 50
    
    # Validate cache size
    if args.cache_size < 5:
        logging.warning(f"Cache size {args.cache_size} is very low, setting to 5 minimum")
        args.cache_size = 5

def signal_handler(signum: int, frame: Optional[object]) -> None:
    """
    Handle shutdown signals gracefully.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logging.info(f"Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)

async def async_main(args: argparse.Namespace) -> None:
    """
    Run the application server asynchronously with command-line options.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        # Validate arguments
        validate_args(args)
        
        # Create and configure the application with command-line arguments
        app = await async_create_app(
            memory_limit_mb=args.memory_limit,
            cache_size=args.cache_size,
            disable_custom_templates=args.disable_custom_templates,
            simple_charts=args.simple_charts
        )
        
        # Log final metrics before starting the server
        metrics = performance_monitor.get_metrics()
        logging.info("Final performance metrics:")
        for name, value in metrics.items():
            logging.info(f"{name}: {value:.2f}")
        
        # Run the server with the specified options
        app.run_server(
            debug=args.debug,
            port=args.port,
            host=args.host,
            use_reloader=args.debug,
            processes=args.workers
        )
        
    except Exception as e:
        logging.error(f"Error running application: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

def main() -> None:
    """Run the application server with command-line options"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Set up signal handlers for graceful shutdown
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the application
        if sys.platform == 'win32':
            # On Windows, use synchronous app creation
            app = create_app(
                memory_limit_mb=args.memory_limit,
                cache_size=args.cache_size,
                disable_custom_templates=args.disable_custom_templates,
                simple_charts=args.simple_charts
            )
            app.run_server(
                debug=args.debug,
                port=args.port,
                host=args.host,
                use_reloader=args.debug,
                processes=args.workers
            )
        else:
            # On Unix-like systems, use async app creation
            asyncio.run(async_main(args))
            
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error running application: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 