"""
Cache implementation for storing filtered dataframes with advanced memory management
"""
import time
import logging
import sys
import gc
import threading
import asyncio
from typing import Dict, Any, Optional, Tuple, Union, TypeVar, Generic, List, Set, cast
from typing_extensions import Protocol
from contextlib import contextmanager
import pandas as pd
import hashlib
import json
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Hashable(Protocol):
    """Protocol for hashable objects"""
    def __hash__(self) -> int: ...

class AsyncLock:
    """
    A lock that works in both synchronous and asynchronous contexts.
    """
    def __init__(self) -> None:
        self._thread_lock = threading.RLock()
        self._async_lock = asyncio.Lock()
    
    @contextmanager
    def __call__(self) -> 'AsyncLock':
        """
        Acquire the lock in a context manager for synchronous code.
        """
        self._thread_lock.acquire()
        try:
            yield self
        finally:
            self._thread_lock.release()
    
    async def __aenter__(self) -> 'AsyncLock':
        """
        Acquire the lock in an async context manager.
        """
        await self._async_lock.acquire()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Release the lock in an async context manager.
        """
        self._async_lock.release()

class DataFrameCache:
    """
    Advanced cache for storing filtered dataframes with memory optimization and performance monitoring.
    Uses LRU strategy with adaptive memory management and parallel processing capabilities.
    """
    def __init__(self, max_size: int = 10, max_memory_mb: Optional[int] = 500):
        """
        Initialize the cache with memory monitoring and optimization
        
        Args:
            max_size: Maximum number of dataframes to store in cache
            max_memory_mb: Maximum memory usage in MB, None for no limit
        """
        self.cache: Dict[str, pd.DataFrame] = {}
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self.timestamps: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lock = AsyncLock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        # Keep track of access frequency for more intelligent eviction
        self.access_count: Dict[str, int] = {}
        self._initialize_memory_monitoring()
    
    def _initialize_memory_monitoring(self) -> None:
        """Initialize memory monitoring and optimization settings"""
        try:
            system_memory = psutil.virtual_memory().total
            # Allow up to 10% of system memory or specified limit, whichever is smaller
            self.memory_threshold = min(system_memory * 0.1, self.max_memory_bytes or float('inf'))
            logger.info(f"Memory threshold set to {self.memory_threshold / (1024**2):.1f} MB")
            
            # Schedule periodic memory checks
            self._start_memory_monitor()
        except Exception as e:
            logger.warning(f"Could not initialize memory monitoring: {e}")
            self.memory_threshold = self.max_memory_bytes or float('inf')
    
    def _start_memory_monitor(self) -> None:
        """Start a background thread to monitor memory usage"""
        def monitor_memory() -> None:
            while True:
                try:
                    # Check memory usage every 30 seconds
                    time.sleep(30)
                    current_memory = self._estimate_memory_usage()
                    if current_memory > self.memory_threshold * 0.9:  # If at 90% of threshold
                        logger.info(f"Memory usage high ({current_memory / (1024**2):.1f} MB), running cleanup")
                        self._reduce_memory_usage(int(current_memory * 0.2))  # Try to free 20% of current usage
                    
                    # Run garbage collection occasionally
                    if time.time() % 120 < 1:  # Approximately every 2 minutes
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error in memory monitor: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start as daemon thread so it doesn't block process exit
        thread = threading.Thread(target=monitor_memory, daemon=True)
        thread.start()
    
    @lru_cache(maxsize=1000)
    def get_key(self, filters: Tuple) -> str:
        """
        Create a hashable key from the filters with caching
        
        Args:
            filters: A tuple of filter parameters
            
        Returns:
            A hashable key
        """
        try:
            # Use a more robust hashing algorithm
            hash_str = json.dumps(str(filters), sort_keys=True).encode()
            return hashlib.blake2b(hash_str, digest_size=16).hexdigest()
        except Exception as e:
            logger.warning(f"Error creating hash for filters: {e}")
            return str(hash(str(filters)))
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        
        Args:
            df: The DataFrame to optimize
            
        Returns:
            The optimized DataFrame
        """
        try:
            # Create a copy to avoid modifying the original
            df_opt = df.copy()
            
            # Optimize numeric columns
            for col in df_opt.select_dtypes(include=['float64']).columns:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
            
            for col in df_opt.select_dtypes(include=['int64']).columns:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
            
            # Optimize string columns
            for col in df_opt.select_dtypes(include=['object']).columns:
                if df_opt[col].nunique() / len(df_opt) < 0.5:  # If less than 50% unique values
                    df_opt[col] = df_opt[col].astype('category')
            
            # Use sparse arrays for data with many duplicates
            for col in df_opt.select_dtypes(include=['float']).columns:
                # If more than 25% of values are the same, use sparse representation
                if (df_opt[col].value_counts().iloc[0] / len(df_opt)) > 0.25:
                    try:
                        df_opt[col] = df_opt[col].astype(pd.SparseDtype(df_opt[col].dtype))
                    except Exception:
                        pass  # Skip if sparse conversion fails
            
            # Force garbage collection after optimization
            gc.collect()
            
            return df_opt
        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return df
    
    def get(self, filters: Tuple) -> Optional[pd.DataFrame]:
        """
        Retrieve a dataframe from cache with memory optimization
        
        Args:
            filters: A tuple of filter parameters
            
        Returns:
            The cached dataframe or None if not found
        """
        with self.lock():
            key = self.get_key(filters)
            if key in self.cache:
                self.timestamps[key] = time.time()
                self.hits += 1
                # Increment access count for this key
                self.access_count[key] = self.access_count.get(key, 0) + 1
                logger.debug(f"Cache hit for filter set {key}")
                return self.cache[key].copy()
            
            self.misses += 1
            return None
    
    async def get_async(self, filters: Tuple) -> Optional[pd.DataFrame]:
        """
        Asynchronously retrieve a dataframe from cache
        
        Args:
            filters: A tuple of filter parameters
            
        Returns:
            The cached dataframe or None if not found
        """
        async with self.lock:
            key = self.get_key(filters)
            if key in self.cache:
                self.timestamps[key] = time.time()
                self.hits += 1
                # Increment access count for this key
                self.access_count[key] = self.access_count.get(key, 0) + 1
                logger.debug(f"Cache hit for filter set {key}")
                return self.cache[key].copy()
            
            self.misses += 1
            return None
    
    def set(self, filters: Tuple, df: pd.DataFrame) -> None:
        """
        Store a dataframe in the cache with memory optimization
        
        Args:
            filters: A tuple of filter parameters
            df: The dataframe to cache
        """
        with self.lock():
            key = self.get_key(filters)
            
            # Optimize DataFrame before caching
            df_optimized = self.optimize_dataframe(df)
            
            # Check memory usage
            if self.max_memory_bytes:
                current_memory = self._estimate_memory_usage()
                new_df_memory = sys.getsizeof(df_optimized)
                
                if current_memory + new_df_memory > self.max_memory_bytes:
                    self._reduce_memory_usage(new_df_memory)
            
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Add new item to cache
            self.cache[key] = df_optimized
            self.timestamps[key] = time.time()
            # Initialize access count
            self.access_count[key] = 1
            logger.debug(f"Added optimized dataframe to cache with key {key}")
    
    async def set_async(self, filters: Tuple, df: pd.DataFrame) -> None:
        """
        Asynchronously store a dataframe in the cache with memory optimization
        
        Args:
            filters: A tuple of filter parameters
            df: The dataframe to cache
        """
        async with self.lock:
            key = self.get_key(filters)
            
            # Optimize DataFrame before caching - run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            df_optimized = await loop.run_in_executor(
                self.executor, self.optimize_dataframe, df
            )
            
            # Check memory usage
            if self.max_memory_bytes:
                current_memory = self._estimate_memory_usage()
                new_df_memory = sys.getsizeof(df_optimized)
                
                if current_memory + new_df_memory > self.max_memory_bytes:
                    await loop.run_in_executor(
                        self.executor, self._reduce_memory_usage, new_df_memory
                    )
            
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.max_size and key not in self.cache:
                await loop.run_in_executor(self.executor, self._evict_lru)
            
            # Add new item to cache
            self.cache[key] = df_optimized
            self.timestamps[key] = time.time()
            # Initialize access count
            self.access_count[key] = 1
            logger.debug(f"Added optimized dataframe to cache with key {key}")
    
    def _evict_lru(self) -> None:
        """
        Remove the least recently used item from the cache.
        Uses a weighted score of recency and frequency.
        """
        if not self.cache:
            return
        
        # Combine recency (timestamp) and frequency (access count) for eviction decision
        current_time = time.time()
        eviction_scores = {}
        
        for key in self.cache:
            # Calculate time factor (0 to 1, where 1 is oldest)
            time_factor = 1.0 - min(1.0, (current_time - self.timestamps[key]) / 3600)
            # Calculate frequency factor (0 to 1, where 1 is most frequent)
            freq_factor = self.access_count.get(key, 0) / (max(self.access_count.values()) + 1)
            # Combined score - higher score means more valuable (less likely to evict)
            eviction_scores[key] = (0.7 * time_factor) + (0.3 * freq_factor)
        
        # Find the key with the lowest score
        lru_key = min(eviction_scores, key=eviction_scores.get)
        
        # Remove the item
        del self.cache[lru_key]
        del self.timestamps[lru_key]
        if lru_key in self.access_count:
            del self.access_count[lru_key]
        
        self.evictions += 1
        logger.debug(f"Evicted cache entry with key {lru_key}")
        
        # Force garbage collection after eviction
        gc.collect()
    
    def _reduce_memory_usage(self, required_bytes: int) -> None:
        """
        Reduce memory usage by removing cached items.
        
        Args:
            required_bytes: Number of bytes that need to be freed
        """
        if not self.cache:
            return
        
        # Calculate memory usage for each cached dataframe
        memory_usage = {}
        for key, df in self.cache.items():
            memory_usage[key] = sys.getsizeof(df)
        
        # Sort by memory usage (largest first) and timestamps (oldest first)
        # This prioritizes removing large, old items
        sorted_keys = sorted(memory_usage.keys(), 
                           key=lambda k: (
                               -memory_usage[k],  # Negative to sort largest first
                               -self.timestamps[k]  # Negative to sort oldest first
                           ))
        
        freed_bytes = 0
        removed_keys = []
        
        # Remove items until we've freed enough memory
        for key in sorted_keys:
            if freed_bytes >= required_bytes:
                break
            
            freed_bytes += memory_usage[key]
            removed_keys.append(key)
        
        # Actually remove the items
        for key in removed_keys:
            del self.cache[key]
            del self.timestamps[key]
            if key in self.access_count:
                del self.access_count[key]
            
            self.evictions += 1
            logger.debug(f"Removed cache entry with key {key} to free memory")
        
        # Force garbage collection after memory reduction
        gc.collect()
        
        logger.info(f"Memory reduction complete. Freed approximately {freed_bytes / (1024*1024):.1f} MB by removing {len(removed_keys)} items")
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate current memory usage of the cache in bytes.
        
        Returns:
            Estimated memory usage in bytes
        """
        total_size = sum(sys.getsizeof(df) for df in self.cache.values())
        return total_size
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock():
            self.cache.clear()
            self.timestamps.clear()
            self.access_count.clear()
            gc.collect()
            logger.info("Cache cleared")
    
    async def clear_async(self) -> None:
        """Asynchronously clear the entire cache."""
        async with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_count.clear()
            
            # Run garbage collection in thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, gc.collect)
            
            logger.info("Cache cleared asynchronously")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics
        
        Returns:
            Dictionary of cache statistics
        """
        with self.lock():
            hit_rate = self.hits / max(1, (self.hits + self.misses)) * 100
            memory_usage_mb = self._estimate_memory_usage() / (1024 * 1024)
            avg_size_mb = memory_usage_mb / max(1, len(self.cache))
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': hit_rate,
                'evictions': self.evictions,
                'memory_usage_mb': memory_usage_mb,
                'avg_entry_size_mb': avg_size_mb,
                'uptime_seconds': time.time() - min(self.timestamps.values()) if self.timestamps else 0
            }
    
    async def get_stats_async(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics asynchronously
        
        Returns:
            Dictionary of cache statistics
        """
        async with self.lock:
            hit_rate = self.hits / max(1, (self.hits + self.misses)) * 100
            memory_usage_mb = self._estimate_memory_usage() / (1024 * 1024)
            avg_size_mb = memory_usage_mb / max(1, len(self.cache))
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': hit_rate,
                'evictions': self.evictions,
                'memory_usage_mb': memory_usage_mb,
                'avg_entry_size_mb': avg_size_mb,
                'uptime_seconds': time.time() - min(self.timestamps.values()) if self.timestamps else 0
            }
    
    def cache_method(self, func):
        """
        Decorator for caching method results.
        
        Args:
            func: The function to cache
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            # Create a key from args and kwargs
            key = (func.__name__,) + args + tuple(sorted(kwargs.items()))
            
            # Try to get from cache
            result = self.get(key)
            if result is not None:
                return result
            
            # Not in cache, compute the result
            result = func(instance, *args, **kwargs)
            
            # Only cache dataframes
            if isinstance(result, pd.DataFrame):
                self.set(key, result)
            
            return result
        
        return wrapper 