"""Tests for the pandas DataFrame accessor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import queryframe.core.accessor as accessor_mod
from queryframe.core.accessor import (
    QueryFrameAccessor,
    _get_engine,
    ask,
    configure,
)
from queryframe.core.result import QueryResult


@pytest.fixture(autouse=True)
def reset_global_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset _global_engine to None before each test."""
    monkeypatch.setattr(accessor_mod, "_global_engine", None)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


class TestConfigure:
    @patch("queryframe.core.accessor.QueryEngine")
    @patch("queryframe.core.accessor.QueryFrameConfig")
    def test_configure_creates_new_engine(
        self, mock_config_cls: MagicMock, mock_engine_cls: MagicMock
    ) -> None:
        mock_cfg = MagicMock()
        mock_config_cls.from_env.return_value.with_overrides.return_value = mock_cfg

        configure(provider="openai", model="gpt-4o")

        mock_config_cls.from_env.assert_called_once()
        mock_config_cls.from_env.return_value.with_overrides.assert_called_once_with(
            provider="openai", model="gpt-4o"
        )
        mock_engine_cls.assert_called_once_with(config=mock_cfg)
        assert accessor_mod._global_engine is mock_engine_cls.return_value

    @patch("queryframe.core.accessor.QueryEngine")
    @patch("queryframe.core.accessor.QueryFrameConfig")
    def test_configure_replaces_existing_engine(
        self, mock_config_cls: MagicMock, mock_engine_cls: MagicMock
    ) -> None:
        configure(provider="openai")
        first_engine = accessor_mod._global_engine

        configure(provider="anthropic")
        second_engine = accessor_mod._global_engine

        assert mock_engine_cls.call_count == 2
        # The engines are both MagicMock return values so identity check isn't meaningful,
        # but call count confirms two engines were created.


class TestGetEngine:
    @patch("queryframe.core.accessor.QueryEngine")
    def test_lazy_creation(self, mock_engine_cls: MagicMock) -> None:
        engine = _get_engine()
        mock_engine_cls.assert_called_once()
        assert engine is mock_engine_cls.return_value

    @patch("queryframe.core.accessor.QueryEngine")
    def test_returns_same_engine(self, mock_engine_cls: MagicMock) -> None:
        engine1 = _get_engine()
        engine2 = _get_engine()
        assert engine1 is engine2
        mock_engine_cls.assert_called_once()


class TestAskFunction:
    @patch("queryframe.core.accessor.QueryEngine")
    def test_ask_delegates_to_engine(
        self, mock_engine_cls: MagicMock, sample_df: pd.DataFrame
    ) -> None:
        expected = MagicMock(spec=QueryResult)
        mock_engine_cls.return_value.ask.return_value = expected

        result = ask(sample_df, "average of a?")

        mock_engine_cls.return_value.ask.assert_called_once_with(
            sample_df, "average of a?"
        )
        assert result is expected

    @patch("queryframe.core.accessor.QueryEngine")
    def test_ask_passes_kwargs(
        self, mock_engine_cls: MagicMock, sample_df: pd.DataFrame
    ) -> None:
        ask(sample_df, "q", viz="matplotlib")
        mock_engine_cls.return_value.ask.assert_called_once_with(
            sample_df, "q", viz="matplotlib"
        )


class TestQueryFrameAccessor:
    def test_accessor_registered(self, sample_df: pd.DataFrame) -> None:
        """The .qf accessor should be available on DataFrames."""
        assert hasattr(sample_df, "qf")
        assert isinstance(sample_df.qf, QueryFrameAccessor)

    @patch("queryframe.core.accessor.QueryEngine")
    def test_accessor_ask(
        self, mock_engine_cls: MagicMock, sample_df: pd.DataFrame
    ) -> None:
        expected = MagicMock(spec=QueryResult)
        mock_engine_cls.return_value.ask.return_value = expected

        result = sample_df.qf.ask("sum of b?")

        mock_engine_cls.return_value.ask.assert_called_once_with(
            sample_df, "sum of b?"
        )
        assert result is expected

    @patch("queryframe.core.accessor.QueryFrameConfig")
    @patch("queryframe.core.accessor.QueryEngine")
    def test_accessor_config(
        self,
        mock_engine_cls: MagicMock,
        mock_config_cls: MagicMock,
        sample_df: pd.DataFrame,
    ) -> None:
        mock_config_cls.from_env.return_value.with_overrides.return_value = MagicMock()
        sample_df.qf.config(provider="anthropic")
        mock_config_cls.from_env.assert_called_once()
