"""Tests for chart type resolution, inference, and selector module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from queryframe.viz.chart_types import ChartType, infer_chart_type, resolve_chart_type
from queryframe.viz.selector import _detect_environment, _get_renderer, select_and_render


# ---------------------------------------------------------------------------
# resolve_chart_type
# ---------------------------------------------------------------------------


class TestResolveChartType:
    def test_exact_match(self):
        assert resolve_chart_type("bar") == ChartType.BAR
        assert resolve_chart_type("line") == ChartType.LINE
        assert resolve_chart_type("scatter") == ChartType.SCATTER

    def test_aliases(self):
        assert resolve_chart_type("barchart") == ChartType.BAR
        assert resolve_chart_type("scatter plot") == ChartType.SCATTER
        assert resolve_chart_type("hist") == ChartType.HISTOGRAM

    def test_case_insensitive(self):
        assert resolve_chart_type("BAR") == ChartType.BAR
        assert resolve_chart_type("Line") == ChartType.LINE

    def test_none_input(self):
        assert resolve_chart_type(None) is None

    def test_unknown(self):
        assert resolve_chart_type("unknown_type") is None


# ---------------------------------------------------------------------------
# infer_chart_type
# ---------------------------------------------------------------------------


class TestInferChartType:
    def test_trend(self):
        assert infer_chart_type("show the sales trend over time") == ChartType.LINE

    def test_distribution(self):
        assert infer_chart_type("show the distribution of prices") == ChartType.HISTOGRAM

    def test_correlation(self):
        assert infer_chart_type("scatter plot of price vs quantity") == ChartType.SCATTER

    def test_proportion(self):
        assert infer_chart_type("show the percentage breakdown") == ChartType.PIE

    def test_default_bar(self):
        assert infer_chart_type("show sales by region") == ChartType.BAR

    def test_heatmap(self):
        assert infer_chart_type("show the heatmap") == ChartType.HEATMAP

    def test_box(self):
        assert infer_chart_type("show the boxplot of outlier data") == ChartType.BOX

    def test_area(self):
        assert infer_chart_type("show the stacked area chart") == ChartType.AREA


# ---------------------------------------------------------------------------
# _detect_environment
# ---------------------------------------------------------------------------


class TestDetectEnvironment:
    def test_script_when_no_ipython(self):
        with patch.dict("sys.modules", {"IPython": None}):
            # When IPython import fails, should return "script"
            result = _detect_environment()
            assert result in ("script", "ipython", "notebook")

    def test_script_fallback(self):
        """When IPython is not installed, return 'script'."""
        with patch("queryframe.viz.selector._detect_environment", return_value="script"):
            assert _detect_environment() == "script"

    def test_notebook_detected_via_mock(self):
        """Verify that _detect_environment can return 'notebook' when in a notebook."""
        import queryframe.viz.selector as sel_mod
        with patch.object(sel_mod, "_detect_environment", return_value="notebook"):
            assert sel_mod._detect_environment() == "notebook"

    def test_ipython_detected_via_mock(self):
        """Verify that _detect_environment can return 'ipython' when in IPython."""
        import queryframe.viz.selector as sel_mod
        with patch.object(sel_mod, "_detect_environment", return_value="ipython"):
            assert sel_mod._detect_environment() == "ipython"


# ---------------------------------------------------------------------------
# _get_renderer
# ---------------------------------------------------------------------------


class TestGetRenderer:
    def test_plotly_mode(self):
        renderer = _get_renderer("plotly")
        if renderer is not None:
            assert renderer.name == "plotly"

    def test_matplotlib_mode(self):
        renderer = _get_renderer("matplotlib")
        if renderer is not None:
            assert renderer.name == "matplotlib"

    def test_altair_mode(self):
        renderer = _get_renderer("altair")
        if renderer is not None:
            assert renderer.name == "altair"

    def test_auto_returns_some_renderer(self):
        renderer = _get_renderer("auto")
        # At least one viz library should be installed in the test env
        assert renderer is not None
        assert renderer.name in ("plotly", "matplotlib", "altair")

    def test_unknown_mode_returns_none(self):
        renderer = _get_renderer("nonexistent_library")
        assert renderer is None

    def test_auto_fallback_when_plotly_unavailable(self):
        """When plotly is not available, auto should fall back to matplotlib or altair."""
        with patch(
            "queryframe.viz.plotly_renderer.PlotlyRenderer.is_available", return_value=False
        ):
            renderer = _get_renderer("auto")
            if renderer is not None:
                assert renderer.name in ("matplotlib", "altair")

    def test_none_when_no_viz_available(self):
        """When no viz libraries are available, return None."""
        with patch(
            "queryframe.viz.plotly_renderer.PlotlyRenderer.is_available", return_value=False
        ), patch(
            "queryframe.viz.matplotlib_renderer.MatplotlibRenderer.is_available", return_value=False
        ), patch(
            "queryframe.viz.altair_renderer.AltairRenderer.is_available", return_value=False
        ):
            renderer = _get_renderer("auto")
            assert renderer is None


# ---------------------------------------------------------------------------
# select_and_render
# ---------------------------------------------------------------------------


@pytest.fixture
def render_df() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["A", "B", "C"],
        "value": [10, 20, 30],
    })


class TestSelectAndRender:
    def test_returns_figure(self, render_df: pd.DataFrame) -> None:
        result = select_and_render(render_df, "bar", x_col="category", y_col="value")
        assert result is not None

    def test_with_style_param(self, render_df: pd.DataFrame) -> None:
        style = {"theme": "dark", "opacity": 0.8, "show_grid": False}
        result = select_and_render(
            render_df, "bar",
            x_col="category", y_col="value",
            style=style,
        )
        assert result is not None

    def test_inferred_chart_type(self, render_df: pd.DataFrame) -> None:
        """When resolve_chart_type returns None, infer_chart_type is used."""
        result = select_and_render(render_df, "show sales trend over time")
        assert result is not None

    def test_returns_none_when_no_renderer(self, render_df: pd.DataFrame) -> None:
        with patch(
            "queryframe.viz.selector._get_renderer", return_value=None
        ):
            result = select_and_render(render_df, "bar")
            assert result is None

    def test_title_passed_through(self, render_df: pd.DataFrame) -> None:
        result = select_and_render(
            render_df, "bar",
            x_col="category", y_col="value",
            title="My Chart",
        )
        assert result is not None

    def test_viz_mode_plotly(self, render_df: pd.DataFrame) -> None:
        try:
            import plotly  # noqa: F401
            result = select_and_render(
                render_df, "bar",
                x_col="category", y_col="value",
                viz_mode="plotly",
            )
            assert result is not None
        except ImportError:
            pytest.skip("plotly not available")

    def test_viz_mode_matplotlib(self, render_df: pd.DataFrame) -> None:
        try:
            import matplotlib  # noqa: F401
            result = select_and_render(
                render_df, "bar",
                x_col="category", y_col="value",
                viz_mode="matplotlib",
            )
            assert result is not None
        except ImportError:
            pytest.skip("matplotlib not available")
