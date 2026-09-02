import pandas as pd
import pytest

from src.data.data_loader import create_sample_data, validate_schema


def test_validate_schema_reports_missing_required_columns():
    incomplete = pd.DataFrame({"title": ["Example"]})

    with pytest.raises(ValueError, match="console"):
        validate_schema(incomplete)


def test_sample_data_respects_cache_configuration():
    _, cache = create_sample_data(cache_size=3, memory_limit_mb=25)

    assert cache.max_size == 3
    assert cache.max_memory_bytes == 25 * 1024 * 1024
