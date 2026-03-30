"""Integration tests for the query engine with mock provider."""

import pandas as pd
import pytest

from queryframe.core.config import QueryFrameConfig
from queryframe.core.engine import QueryEngine


class TestQueryEngine:
    def test_basic_query(self, sample_df, mock_provider):
        engine = QueryEngine(
            config=QueryFrameConfig(cache_enabled=False),
            provider=mock_provider,
        )
        result = engine.ask(sample_df, "how many rows?")

        assert result.data == 8  # df.shape[0] = 8
        assert result.query == "how many rows?"
        assert result.code == "result = df.shape[0]"
        assert result.provider == "mock"
        assert result.latency_ms > 0

    def test_caching(self, sample_df, mock_provider):
        engine = QueryEngine(
            config=QueryFrameConfig(cache_enabled=True),
            provider=mock_provider,
        )

        result1 = engine.ask(sample_df, "how many rows?")
        result2 = engine.ask(sample_df, "how many rows?")

        assert not result1.cached
        assert result2.cached
        assert result2.latency_ms < result1.latency_ms

    def test_clear_cache(self, sample_df, mock_provider):
        engine = QueryEngine(
            config=QueryFrameConfig(cache_enabled=True),
            provider=mock_provider,
        )

        engine.ask(sample_df, "how many rows?")
        engine.clear_cache()
        result = engine.ask(sample_df, "how many rows?")
        assert not result.cached

    def test_clear_memory(self, sample_df, mock_provider):
        engine = QueryEngine(
            config=QueryFrameConfig(cache_enabled=False),
            provider=mock_provider,
        )

        engine.ask(sample_df, "first query")
        assert len(engine._conversation) == 1
        engine.clear_memory()
        assert len(engine._conversation) == 0

    def test_result_chaining(self, sample_df, mock_provider):
        engine = QueryEngine(
            config=QueryFrameConfig(cache_enabled=False),
            provider=mock_provider,
        )

        result = engine.ask(sample_df, "how many rows?")
        # Verify the result has engine reference for chaining
        assert result._engine is engine
        assert result._df is sample_df


class TestQueryEngineSandbox:
    def test_unsafe_code_rejected(self, sample_df):
        """Provider returns unsafe code — engine should handle gracefully."""
        from tests.conftest import MockProvider

        unsafe_provider = MockProvider(
            response_content='{"code": "import os; result = os.listdir()", "explanation": "list files"}'
        )
        engine = QueryEngine(
            config=QueryFrameConfig(cache_enabled=False, max_retries=0),
            provider=unsafe_provider,
        )

        result = engine.ask(sample_df, "list files")
        # Should fail gracefully, not crash
        assert result.explanation.startswith("Error:")
