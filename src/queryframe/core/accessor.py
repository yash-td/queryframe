"""Pandas DataFrame accessor — enables df.qf.ask() and df.ask()."""

from __future__ import annotations

from typing import Any

import pandas as pd

from queryframe.core.config import QueryFrameConfig
from queryframe.core.engine import QueryEngine
from queryframe.core.result import QueryResult

# Global engine instance (lazily created)
_global_engine: QueryEngine | None = None


def _get_engine() -> QueryEngine:
    """Get or create the global engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = QueryEngine()
    return _global_engine


def configure(**kwargs: Any) -> None:
    """Configure the global QueryFrame engine.

    Example:
        import queryframe as qf
        qf.configure(provider="openai", model="gpt-4o")
    """
    global _global_engine
    config = QueryFrameConfig.from_env().with_overrides(**kwargs)
    _global_engine = QueryEngine(config=config)


def ask(df: pd.DataFrame, query: str, **kwargs: Any) -> QueryResult:
    """Ask a question about a DataFrame using the global engine.

    Example:
        import queryframe as qf
        result = qf.ask(df, "what is the average sales by region?")
    """
    return _get_engine().ask(df, query, **kwargs)


@pd.api.extensions.register_dataframe_accessor("qf")
class QueryFrameAccessor:
    """Pandas accessor that adds .qf.ask() to DataFrames.

    Example:
        result = df.qf.ask("show me sales by region")
        result = df.qf.ask("what is the average price?")
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._df = pandas_obj

    def ask(self, query: str, **kwargs: Any) -> QueryResult:
        """Ask a natural language question about this DataFrame."""
        return _get_engine().ask(self._df, query, **kwargs)

    def config(self, **kwargs: Any) -> None:
        """Configure the global engine."""
        configure(**kwargs)
