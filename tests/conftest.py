"""Shared test fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pandas as pd
import pytest

from queryframe.llm.base import LLMProvider, LLMResponse, TokenUsage


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A sample sales DataFrame for testing."""
    return pd.DataFrame({
        "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
        "product": ["Widget", "Widget", "Widget", "Widget", "Gadget", "Gadget", "Gadget", "Gadget"],
        "sales": [100, 150, 200, 120, 80, 90, 110, 95],
        "quantity": [10, 15, 20, 12, 8, 9, 11, 10],
        "date": pd.date_range("2024-01-01", periods=8, freq="MS"),
    })


@pytest.fixture
def wide_df() -> pd.DataFrame:
    """A wide DataFrame with 60 columns for compression testing."""
    import numpy as np

    data = {f"col_{i}": np.random.randn(100) for i in range(60)}
    data["category"] = [f"cat_{i % 5}" for i in range(100)]
    return pd.DataFrame(data)


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """A purely numeric DataFrame."""
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
        "z": [1, 3, 5, 7, 9],
    })


class MockProvider:
    """Mock LLM provider that returns canned responses."""

    def __init__(self, response_content: str = '{"code": "result = df.shape[0]", "chart_type": null, "explanation": "Row count"}'):
        self._response_content = response_content

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        return LLMResponse(
            content=self._response_content,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="mock-model",
            latency_ms=10.0,
        )


@pytest.fixture
def mock_provider() -> MockProvider:
    """A mock LLM provider."""
    return MockProvider()


@pytest.fixture
def viz_mock_provider() -> MockProvider:
    """A mock provider that returns visualization code."""
    return MockProvider(
        response_content='{"code": "result = df.groupby(\'region\')[\'sales\'].sum().reset_index()", '
        '"chart_type": "bar", "x_col": "region", "y_col": "sales", '
        '"title": "Sales by Region", "explanation": "Bar chart of sales by region"}'
    )
