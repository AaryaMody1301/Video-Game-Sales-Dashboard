import pandas as pd
import pytest

from src.utils.exporting import serialize_dataframe


def test_csv_export_serializes_all_rows():
    df = pd.DataFrame([{"title": "Game A", "total_sales": 1.5}, {"title": "Game B", "total_sales": 2.0}])

    payload, extension = serialize_dataframe(df, "csv")

    assert extension == "csv"
    assert b"Game A" in payload
    assert b"Game B" in payload


def test_excel_export_returns_xlsx_bytes():
    df = pd.DataFrame([{"title": "Game A", "total_sales": 1.5}])

    payload, extension = serialize_dataframe(df, "excel")

    assert extension == "xlsx"
    assert payload.startswith(b"PK")


def test_unsupported_pdf_export_fails_explicitly_instead_of_substituting_csv():
    df = pd.DataFrame([{"title": "Game A"}])

    with pytest.raises(ValueError, match="Unsupported export format"):
        serialize_dataframe(df, "pdf")
