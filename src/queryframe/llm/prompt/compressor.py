"""Schema and prompt compression for token efficiency."""

from __future__ import annotations

from queryframe.core.schema import SchemaInfo, compress_schema


def compress_for_local(
    schema: SchemaInfo,
    query: str,
    max_columns: int = 30,
) -> str:
    """Compress schema aggressively for local models with smaller context windows.

    Reduces sample values, abbreviates dtypes, limits column count.
    """
    columns = schema.columns

    # Filter to relevant columns for wide DataFrames
    if len(columns) > max_columns:
        query_lower = query.lower()
        relevant = [c for c in columns if c.name.lower() in query_lower]
        remaining = [c for c in columns if c not in relevant][:8]
        columns = tuple(relevant + remaining) if relevant else columns[:max_columns]

    # Abbreviate dtype names
    dtype_map = {
        "int64": "int",
        "float64": "float",
        "object": "str",
        "bool": "bool",
        "datetime64[ns]": "datetime",
        "category": "cat",
        "Int64": "int",
        "Float64": "float",
        "string": "str",
    }

    lines = [f"DF: {schema.shape[0]}x{schema.shape[1]}"]
    for col in columns:
        dtype = dtype_map.get(col.dtype, col.dtype)
        parts = [f"  {col.name}({dtype})"]
        if col.sample_values:
            parts.append(f" ex:[{col.sample_values[0]}]")
        lines.append("".join(parts))

    return "\n".join(lines)
