"""Auto-select the best visualization library and render charts."""

from __future__ import annotations

import os
from typing import Any

from queryframe.utils.logger import get_logger
from queryframe.viz.chart_types import ChartType, resolve_chart_type, infer_chart_type

logger = get_logger(__name__)


def _detect_environment() -> str:
    """Detect the execution environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell_name = type(shell).__name__
            if "ZMQ" in shell_name or "Kernel" in shell_name:
                return "notebook"
            return "ipython"
    except (ImportError, NameError):
        pass
    return "script"


def _get_renderer(viz_mode: str) -> Any:
    """Get the appropriate renderer based on mode and availability."""
    if viz_mode == "plotly":
        from queryframe.viz.plotly_renderer import PlotlyRenderer
        if PlotlyRenderer.is_available():
            return PlotlyRenderer()

    if viz_mode == "matplotlib":
        from queryframe.viz.matplotlib_renderer import MatplotlibRenderer
        if MatplotlibRenderer.is_available():
            return MatplotlibRenderer()

    if viz_mode == "altair":
        from queryframe.viz.altair_renderer import AltairRenderer
        if AltairRenderer.is_available():
            return AltairRenderer()

    if viz_mode == "auto":
        env = _detect_environment()

        # In notebooks, prefer Plotly for interactivity
        if env == "notebook":
            from queryframe.viz.plotly_renderer import PlotlyRenderer
            if PlotlyRenderer.is_available():
                return PlotlyRenderer()

        # Try Plotly first (best interactive experience)
        try:
            from queryframe.viz.plotly_renderer import PlotlyRenderer
            if PlotlyRenderer.is_available():
                return PlotlyRenderer()
        except Exception:
            pass

        # Fall back to Matplotlib
        try:
            from queryframe.viz.matplotlib_renderer import MatplotlibRenderer
            if MatplotlibRenderer.is_available():
                return MatplotlibRenderer()
        except Exception:
            pass

        # Fall back to Altair
        try:
            from queryframe.viz.altair_renderer import AltairRenderer
            if AltairRenderer.is_available():
                return AltairRenderer()
        except Exception:
            pass

    return None


def select_and_render(
    data: Any,
    chart_type: str,
    x_col: str | None = None,
    y_col: str | None = None,
    title: str | None = None,
    viz_mode: str = "auto",
    style: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Select the best renderer and render a chart.

    Args:
        data: DataFrame or data to visualize
        chart_type: Chart type string from LLM
        x_col: X-axis column name
        y_col: Y-axis column name
        title: Chart title
        viz_mode: "auto", "plotly", "matplotlib", or "altair"
        style: Raw style dict from LLM (colors, theme, orientation, etc.)

    Returns:
        A figure object (Plotly, Matplotlib, or Altair) or None
    """
    resolved = resolve_chart_type(chart_type)
    if resolved is None:
        resolved = infer_chart_type(chart_type)

    renderer = _get_renderer(viz_mode)
    if renderer is None:
        logger.warning(
            "No visualization library available. "
            "Install one: pip install queryframe[plotly] or pip install queryframe[matplotlib]"
        )
        return None

    logger.info("Using %s renderer for %s chart", renderer.name, resolved.value)

    return renderer.render(
        data=data,
        chart_type=resolved,
        x_col=x_col,
        y_col=y_col,
        title=title,
        style=style,
        **kwargs,
    )
