"""DataFrame schema extraction and compression."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnInfo:
    """Information about a single DataFrame column."""

    name: str
    dtype: str
    n_unique: int
    sample_values: list[str]
    null_pct: float


@dataclass(frozen=True)
class SchemaInfo:
    """Extracted schema information from a DataFrame."""

    columns: tuple[ColumnInfo, ...]
    shape: tuple[int, int]
    fingerprint: str


def extract_schema(df: pd.DataFrame, max_samples: int = 3) -> SchemaInfo:
    """Extract schema information from a DataFrame."""
    columns: list[ColumnInfo] = []
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique())
        null_pct = round(float(series.isna().mean()), 3)

        # Get sample values (non-null, unique)
        non_null = series.dropna().unique()
        samples = [str(v) for v in non_null[:max_samples]]

        columns.append(
            ColumnInfo(
                name=str(col),
                dtype=str(series.dtype),
                n_unique=n_unique,
                sample_values=samples,
                null_pct=null_pct,
            )
        )

    fingerprint = _compute_fingerprint(df)
    return SchemaInfo(
        columns=tuple(columns),
        shape=(len(df), len(df.columns)),
        fingerprint=fingerprint,
    )


def _compute_fingerprint(df: pd.DataFrame) -> str:
    """Compute a stable fingerprint for a DataFrame's schema."""
    parts = sorted(f"{col}:{df[col].dtype}" for col in df.columns)
    raw = "|".join(parts) + f"|rows={len(df)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compress_schema(schema: SchemaInfo, query: str, max_columns: int = 50) -> str:
    """Compress schema into a minimal string for LLM prompts.

    For wide DataFrames (>max_columns), only include columns that appear
    relevant to the query based on keyword matching.
    """
    columns = schema.columns

    # Filter to relevant columns for wide DataFrames
    if len(columns) > max_columns:
        query_lower = query.lower()
        relevant = [c for c in columns if c.name.lower() in query_lower]
        # Always include first 10 columns for context
        remaining = [c for c in columns if c not in relevant][:10]
        columns = tuple(relevant + remaining) if relevant else columns[:max_columns]

    lines = [f"DataFrame: {schema.shape[0]} rows x {schema.shape[1]} columns\n"]
    lines.append("Columns:")

    for col in columns:
        parts = [f"  - {col.name} ({col.dtype})"]
        if col.n_unique <= 10 and col.sample_values:
            parts.append(f" unique={col.n_unique}")
            parts.append(f" values=[{', '.join(col.sample_values)}]")
        elif col.sample_values:
            parts.append(f" unique={col.n_unique}")
            parts.append(f" examples=[{', '.join(col.sample_values[:2])}]")
        if col.null_pct > 0:
            parts.append(f" nulls={col.null_pct:.1%}")
        lines.append("".join(parts))

    return "\n".join(lines)
