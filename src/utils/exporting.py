"""Serialization helpers for dashboard exports."""

from __future__ import annotations

from io import BytesIO
from typing import Tuple

import pandas as pd


SUPPORTED_EXPORT_FORMATS = ("csv", "excel")


def serialize_dataframe(df: pd.DataFrame, export_format: str) -> Tuple[bytes, str]:
    """Serialize a dataframe to a supported format and return bytes plus extension."""
    if export_format == "csv":
        return df.to_csv(index=False).encode("utf-8"), "csv"

    if export_format == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="VideoGameSales")
        return output.getvalue(), "xlsx"

    raise ValueError(
        f"Unsupported export format: {export_format}. "
        f"Supported formats: {', '.join(SUPPORTED_EXPORT_FORMATS)}"
    )
