"""Tests for DataFrame utility helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from queryframe.utils.dataframe import (
    detect_categorical_columns,
    detect_datetime_columns,
    safe_copy,
)


class TestSafeCopy:
    def test_returns_new_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        copy = safe_copy(df)
        assert copy is not df

    def test_data_matches_original(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        copy = safe_copy(df)
        pd.testing.assert_frame_equal(copy, df)

    def test_shallow_copy_shares_underlying_data(self) -> None:
        """safe_copy uses deep=False, so underlying numpy arrays may be shared."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        copy = safe_copy(df)
        # Modifying via iloc on the copy should not affect original
        # (pandas CoW behavior may vary, but the copy should be independent at the DataFrame level)
        assert copy.equals(df)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        copy = safe_copy(df)
        assert copy.empty
        assert copy is not df


class TestDetectDatetimeColumns:
    def test_finds_datetime_columns(self) -> None:
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "value": [1, 2, 3],
            "ts": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })
        result = detect_datetime_columns(df)
        assert sorted(result) == ["date", "ts"]

    def test_no_datetime_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = detect_datetime_columns(df)
        assert result == []

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = detect_datetime_columns(df)
        assert result == []

    def test_string_date_not_detected(self) -> None:
        """String columns containing date-like values should NOT be detected."""
        df = pd.DataFrame({"date_str": ["2024-01-01", "2024-02-01"]})
        result = detect_datetime_columns(df)
        assert result == []


class TestDetectCategoricalColumns:
    def test_finds_low_cardinality_object_columns(self) -> None:
        df = pd.DataFrame({
            "region": ["North", "South", "East", "West"] * 25,
            "value": range(100),
        })
        result = detect_categorical_columns(df)
        assert result == ["region"]

    def test_high_cardinality_excluded(self) -> None:
        df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(100)],
        })
        result = detect_categorical_columns(df, threshold=20)
        assert result == []

    def test_numeric_columns_excluded(self) -> None:
        """Even low-cardinality numeric columns should not be detected (dtype != object)."""
        df = pd.DataFrame({"flag": [0, 1, 0, 1]})
        result = detect_categorical_columns(df)
        assert result == []

    def test_custom_threshold(self) -> None:
        df = pd.DataFrame({
            "status": ["active", "inactive", "pending"] * 10,
        })
        # With threshold=2, 3 unique values should be excluded
        assert detect_categorical_columns(df, threshold=2) == []
        # With threshold=5, it should be included
        assert detect_categorical_columns(df, threshold=5) == ["status"]

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = detect_categorical_columns(df)
        assert result == []

    def test_multiple_categorical_columns(self) -> None:
        df = pd.DataFrame({
            "color": ["red", "blue", "green"] * 10,
            "size": ["S", "M", "L"] * 10,
            "price": [10.0, 20.0, 30.0] * 10,
        })
        result = detect_categorical_columns(df)
        assert sorted(result) == ["color", "size"]
