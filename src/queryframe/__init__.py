"""QueryFrame — Super fast natural language data visualization for pandas."""

from queryframe.core.accessor import ask, configure
from queryframe.core.config import QueryFrameConfig
from queryframe.core.engine import QueryEngine
from queryframe.core.result import QueryResult

__version__ = "0.1.0"
__all__ = [
    "QueryEngine",
    "QueryFrameConfig",
    "QueryResult",
    "ask",
    "configure",
]
