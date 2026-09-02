import pandas as pd

from src.utils.cache import DataFrameCache


def test_cache_round_trip_returns_dataframe_copy():
    cache = DataFrameCache(max_size=2, max_memory_mb=50)
    source = pd.DataFrame({"value": [1, 2, 3]})
    filters = ((2000, 2020), ("PC",))

    cache.set(filters, source)
    cached = cache.get(filters)

    assert cached is not None
    pd.testing.assert_frame_equal(cached, source)
    assert cached is not source


def test_cache_evicts_when_capacity_is_exceeded():
    cache = DataFrameCache(max_size=1, max_memory_mb=50)

    cache.set(("first",), pd.DataFrame({"value": [1]}))
    cache.set(("second",), pd.DataFrame({"value": [2]}))

    assert cache.get(("second",)) is not None
    assert cache.evictions >= 1
