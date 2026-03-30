"""DataFrame utility helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Create an efficient copy of a DataFrame for sandbox execution."""
    return df.copy(deep=False)


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain datetime data."""
    import pandas as _pd

    return [
        col
        for col in df.columns
        if _pd.api.types.is_datetime64_any_dtype(df[col])
    ]


def detect_categorical_columns(df: pd.DataFrame, threshold: int = 20) -> list[str]:
    """Return column names likely to be categorical (low cardinality)."""
    return [
        col
        for col in df.columns
        if df[col].nunique() <= threshold and df[col].dtype == "object"
    ]
