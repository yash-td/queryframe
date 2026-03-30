"""Abstract visualization renderer protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from queryframe.viz.chart_types import ChartType


@runtime_checkable
class VizRenderer(Protocol):
    """Protocol that all visualization renderers must implement."""

    @property
    def name(self) -> str:
        """Renderer name (e.g., 'plotly', 'matplotlib')."""
        ...

    def render(
        self,
        data: Any,
        chart_type: ChartType,
        x_col: str | None = None,
        y_col: str | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a chart and return a figure object."""
        ...

    @staticmethod
    def is_available() -> bool:
        """Check if the rendering library is installed."""
        ...
