"""Tests for schema extraction and compression."""

import pandas as pd
import pytest

from queryframe.core.schema import (
    ColumnInfo,
    SchemaInfo,
    compress_schema,
    extract_schema,
    _compute_fingerprint,
)


class TestExtractSchema:
    def test_basic_extraction(self, sample_df):
        schema = extract_schema(sample_df)

        assert schema.shape == (8, 5)
        assert len(schema.columns) == 5
        assert schema.fingerprint  # non-empty

    def test_column_names(self, sample_df):
        schema = extract_schema(sample_df)
        names = [c.name for c in schema.columns]
        assert "region" in names
        assert "sales" in names
        assert "date" in names

    def test_column_dtypes(self, sample_df):
        schema = extract_schema(sample_df)
        col_map = {c.name: c for c in schema.columns}
        assert "int" in col_map["sales"].dtype
        assert col_map["region"].dtype == "object"

    def test_sample_values(self, sample_df):
        schema = extract_schema(sample_df, max_samples=2)
        col_map = {c.name: c for c in schema.columns}
        assert len(col_map["region"].sample_values) <= 2

    def test_null_percentage(self):
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [1, 2, 3, 4]})
        schema = extract_schema(df)
        col_map = {c.name: c for c in schema.columns}
        assert col_map["a"].null_pct == 0.25
        assert col_map["b"].null_pct == 0.0

    def test_fingerprint_stability(self, sample_df):
        """Same DataFrame should produce the same fingerprint."""
        fp1 = _compute_fingerprint(sample_df)
        fp2 = _compute_fingerprint(sample_df)
        assert fp1 == fp2

    def test_fingerprint_changes_with_schema(self, sample_df):
        """Different schemas should produce different fingerprints."""
        fp1 = _compute_fingerprint(sample_df)
        df2 = sample_df.drop(columns=["sales"])
        fp2 = _compute_fingerprint(df2)
        assert fp1 != fp2

    def test_schema_is_frozen(self, sample_df):
        schema = extract_schema(sample_df)
        with pytest.raises(AttributeError):
            schema.shape = (0, 0)


class TestCompressSchema:
    def test_basic_compression(self, sample_df):
        schema = extract_schema(sample_df)
        result = compress_schema(schema, "average sales")
        assert "region" in result
        assert "sales" in result
        assert "8 rows" in result

    def test_wide_dataframe_filtering(self, wide_df):
        schema = extract_schema(wide_df)
        result = compress_schema(schema, "show col_1 values", max_columns=20)
        # Should include relevant columns but not all 61
        assert "col_1" in result

    def test_compression_format(self, sample_df):
        schema = extract_schema(sample_df)
        result = compress_schema(schema, "test")
        assert "DataFrame:" in result
        assert "Columns:" in result
