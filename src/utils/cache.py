"""
Cache implementation for storing filtered dataframes
"""
import time
import logging

logger = logging.getLogger(__name__)

class DataFrameCache:
    """
    Cache for storing filtered dataframes to improve dashboard performance.
    Uses LRU (Least Recently Used) strategy for cache replacement.
    """
    def __init__(self, max_size=10):
        """
        Initialize the cache with a maximum size
        
        Args:
            max_size (int): Maximum number of dataframes to store in cache
        """
        self.cache = {}
        self.max_size = max_size
        self.timestamps = {}
    
    def get_key(self, filters):
        """
        Create a hashable key from the filters
        
        Args:
            filters (tuple): A tuple of filter parameters
            
        Returns:
            str: A hashable key
        """
        return str(hash(str(filters)))
    
    def get(self, filters):
        """
        Retrieve a dataframe from cache if it exists
        
        Args:
            filters (tuple): A tuple of filter parameters
            
        Returns:
            DataFrame: The cached dataframe or None if not found
        """
        key = self.get_key(filters)
        if key in self.cache:
            # Update timestamp for LRU tracking
            self.timestamps[key] = time.time()
            logger.debug(f"Cache hit for filter set {key}")
            return self.cache[key]
        return None
    
    def set(self, filters, df):
        """
        Store a dataframe in the cache
        
        Args:
            filters (tuple): A tuple of filter parameters
            df (DataFrame): The dataframe to cache
        """
        key = self.get_key(filters)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            logger.debug(f"Cache full, removing {oldest_key}")
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Add new item to cache
        self.cache[key] = df.copy()  # Store a copy to avoid reference issues
        self.timestamps[key] = time.time()
        logger.debug(f"Added filtered dataframe to cache with key {key}")
        
    def clear(self):
        """Clear the entire cache"""
        self.cache = {}
        self.timestamps = {}
        logger.debug("Cache cleared") 