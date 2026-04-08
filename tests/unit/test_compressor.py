"""Tests for schema compression and context building."""

from __future__ import annotations

import time

import pytest

from queryframe.core.schema import ColumnInfo, SchemaInfo
from queryframe.llm.prompt.compressor import compress_for_local
from queryframe.memory.context import build_context
from queryframe.memory.conversation import ConversationMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_schema() -> SchemaInfo:
    """A schema with a handful of columns."""
    return SchemaInfo(
        columns=(
            ColumnInfo(name="region", dtype="object", n_unique=4, sample_values=["North", "South"], null_pct=0.0),
            ColumnInfo(name="sales", dtype="int64", n_unique=8, sample_values=["100", "150"], null_pct=0.0),
            ColumnInfo(name="date", dtype="datetime64[ns]", n_unique=8, sample_values=["2024-01-01"], null_pct=0.0),
            ColumnInfo(name="active", dtype="bool", n_unique=2, sample_values=["True"], null_pct=0.05),
            ColumnInfo(name="category", dtype="category", n_unique=3, sample_values=["A"], null_pct=0.0),
        ),
        shape=(100, 5),
        fingerprint="abc123",
    )


@pytest.fixture
def wide_schema() -> SchemaInfo:
    """A schema with >30 columns to trigger column filtering."""
    cols = tuple(
        ColumnInfo(
            name=f"col_{i}",
            dtype="float64",
            n_unique=50,
            sample_values=[f"{i * 1.1:.1f}"],
            null_pct=0.0,
        )
        for i in range(40)
    )
    return SchemaInfo(columns=cols, shape=(1000, 40), fingerprint="wide123")


# ---------------------------------------------------------------------------
# compress_for_local tests
# ---------------------------------------------------------------------------

class TestCompressForLocal:
    def test_header_line(self, normal_schema: SchemaInfo) -> None:
        result = compress_for_local(normal_schema, "show sales")
        assert result.startswith("DF: 100x5")

    def test_dtype_abbreviation(self, normal_schema: SchemaInfo) -> None:
        result = compress_for_local(normal_schema, "query")
        assert "region(str)" in result
        assert "sales(int)" in result
        assert "date(datetime)" in result
        assert "active(bool)" in result
        assert "category(cat)" in result

    def test_sample_values_included(self, normal_schema: SchemaInfo) -> None:
        result = compress_for_local(normal_schema, "query")
        assert "ex:[North]" in result
        assert "ex:[100]" in result

    def test_wide_schema_limits_columns(self, wide_schema: SchemaInfo) -> None:
        result = compress_for_local(wide_schema, "unrelated query", max_columns=30)
        lines = result.strip().split("\n")
        # Header + at most 30 column lines
        assert len(lines) <= 31

    def test_wide_schema_prioritizes_relevant_columns(self) -> None:
        """When query mentions a column name, it should be included."""
        cols = tuple(
            ColumnInfo(
                name=f"col_{i}",
                dtype="float64",
                n_unique=50,
                sample_values=[str(i)],
                null_pct=0.0,
            )
            for i in range(40)
        )
        # Add a column whose name appears in the query
        target = ColumnInfo(
            name="revenue",
            dtype="float64",
            n_unique=100,
            sample_values=["999"],
            null_pct=0.0,
        )
        schema = SchemaInfo(
            columns=cols + (target,),
            shape=(1000, 41),
            fingerprint="x",
        )
        result = compress_for_local(schema, "show me revenue trends", max_columns=10)
        assert "revenue(float)" in result

    def test_no_sample_values(self) -> None:
        schema = SchemaInfo(
            columns=(
                ColumnInfo(name="empty", dtype="object", n_unique=0, sample_values=[], null_pct=1.0),
            ),
            shape=(10, 1),
            fingerprint="e",
        )
        result = compress_for_local(schema, "q")
        assert "empty(str)" in result
        assert "ex:" not in result


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_empty_memory(self) -> None:
        memory = ConversationMemory()
        result = build_context(memory, "fp1")
        assert result == []

    def test_single_turn(self) -> None:
        memory = ConversationMemory()
        memory.add_turn("fp1", query="avg sales?", code="result = df.sales.mean()", result_summary="mean is 120")

        turns = build_context(memory, "fp1")
        assert len(turns) == 1
        assert turns[0].query == "avg sales?"
        # Single turn is treated as the last turn, so code snippet is included
        assert "code:" in turns[0].summary
        assert "result = df.sales.mean()" in turns[0].summary

    def test_multiple_turns(self) -> None:
        memory = ConversationMemory()
        memory.add_turn("fp1", query="q1", code="code1_long_enough_to_show", result_summary="summary1")
        memory.add_turn("fp1", query="q2", code="code2_long_enough_to_show", result_summary="summary2")
        memory.add_turn("fp1", query="q3", code="code3_long_enough_to_show", result_summary="summary3")

        turns = build_context(memory, "fp1", max_turns=3)
        assert len(turns) == 3

        # Older turns: just summary, no code
        assert "code:" not in turns[0].summary
        assert turns[0].summary == "summary1"
        assert "code:" not in turns[1].summary

        # Most recent turn: includes code snippet
        assert "code:" in turns[2].summary

    def test_different_fingerprint_returns_empty(self) -> None:
        memory = ConversationMemory()
        memory.add_turn("fp1", query="q", code="c", result_summary="s")

        turns = build_context(memory, "fp_other")
        assert turns == []

    def test_max_turns_limits_output(self) -> None:
        memory = ConversationMemory()
        for i in range(10):
            memory.add_turn("fp1", query=f"q{i}", code=f"c{i}", result_summary=f"s{i}")

        turns = build_context(memory, "fp1", max_turns=2)
        assert len(turns) == 2
