"""Thread-safe bounded cache for filtered pandas DataFrames."""

from collections import OrderedDict
from functools import lru_cache, wraps
import hashlib
import json
import logging
import threading
from typing import Dict, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataFrameCache:
    """Small LRU cache for repeated dashboard filter results.

    Cache maintenance is performed synchronously during reads and writes. This keeps
    the lifecycle deterministic and avoids creating background threads for every app
    or test instance.
    """

    def __init__(self, max_size: int = 10, max_memory_mb: Optional[int] = 500):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if max_memory_mb is not None and max_memory_mb < 1:
            raise ValueError("max_memory_mb must be at least 1 MB when provided")

        self.cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self.max_size = max_size
        self.max_memory_bytes = (
            max_memory_mb * 1024 * 1024 if max_memory_mb is not None else None
        )
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lock = threading.RLock()

    @lru_cache(maxsize=1000)
    def get_key(self, filters: Tuple) -> str:
        """Return a stable compact key for a filter tuple."""
        payload = json.dumps(filters, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Return a memory-conscious copy without changing dashboard semantics."""
        optimized = df.copy()

        for column in optimized.select_dtypes(include=["float64"]).columns:
            optimized[column] = pd.to_numeric(optimized[column], downcast="float")
        for column in optimized.select_dtypes(include=["int64"]).columns:
            optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
        for column in optimized.select_dtypes(include=["object"]).columns:
            if len(optimized) and optimized[column].nunique(dropna=False) / len(optimized) < 0.5:
                optimized[column] = optimized[column].astype("category")

        return optimized

    @staticmethod
    def _dataframe_bytes(df: pd.DataFrame) -> int:
        return int(df.memory_usage(index=True, deep=True).sum())

    def _estimate_memory_usage(self) -> int:
        return sum(self._dataframe_bytes(df) for df in self.cache.values())

    def _evict_lru(self) -> None:
        if self.cache:
            self.cache.popitem(last=False)
            self.evictions += 1

    def _make_room(self, incoming_bytes: int, replacing_key: Optional[str] = None) -> None:
        while len(self.cache) >= self.max_size and replacing_key not in self.cache:
            self._evict_lru()

        if self.max_memory_bytes is None:
            return

        while self.cache and self._estimate_memory_usage() + incoming_bytes > self.max_memory_bytes:
            self._evict_lru()

    def get(self, filters: Tuple) -> Optional[pd.DataFrame]:
        """Return a copy of a cached DataFrame, or ``None`` on a miss."""
        key = self.get_key(filters)
        with self.lock:
            cached = self.cache.get(key)
            if cached is None:
                self.misses += 1
                return None

            self.cache.move_to_end(key)
            self.hits += 1
            return cached.copy()

    def set(self, filters: Tuple, df: pd.DataFrame) -> None:
        """Store a DataFrame while respecting item and optional memory limits."""
        key = self.get_key(filters)
        optimized = self.optimize_dataframe(df)
        incoming_bytes = self._dataframe_bytes(optimized)

        if self.max_memory_bytes is not None and incoming_bytes > self.max_memory_bytes:
            logger.debug("Skipping cache entry larger than configured memory limit")
            return

        with self.lock:
            if key in self.cache:
                del self.cache[key]
            self._make_room(incoming_bytes, replacing_key=key)
            self.cache[key] = optimized
            self.cache.move_to_end(key)

    def clear(self) -> None:
        """Remove all cached DataFrames."""
        with self.lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Return lightweight cache statistics for diagnostics."""
        with self.lock:
            requests = self.hits + self.misses
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": (self.hits / requests * 100) if requests else 0.0,
                "evictions": self.evictions,
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024),
            }

    def cache_method(self, func):
        """Cache DataFrame results returned by an instance method."""

        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            key = (func.__name__,) + args + tuple(sorted(kwargs.items()))
            cached = self.get(key)
            if cached is not None:
                return cached

            result = func(instance, *args, **kwargs)
            if isinstance(result, pd.DataFrame):
                self.set(key, result)
            return result

        return wrapper
