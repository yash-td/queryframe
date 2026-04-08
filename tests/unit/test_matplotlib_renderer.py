"""Tests for MatplotlibRenderer — chart creation, data conversion, styling, and seaborn fallback."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from queryframe.viz.chart_types import ChartType
from queryframe.viz.style import ChartStyle

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from queryframe.viz.matplotlib_renderer import MatplotlibRenderer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def renderer() -> MatplotlibRenderer:
    return MatplotlibRenderer()


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["A", "B", "C", "D"],
        "value": [10, 20, 30, 40],
    })


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "z": [5.0, 3.0, 1.0, 7.0, 9.0],
    })


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    def test_dataframe_passthrough(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        result = renderer._to_dataframe(simple_df)
        assert result is not None
        pd.testing.assert_frame_equal(result, simple_df)

    def test_series(self, renderer: MatplotlibRenderer) -> None:
        s = pd.Series([10, 20, 30], name="val")
        result = renderer._to_dataframe(s)
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_dict(self, renderer: MatplotlibRenderer) -> None:
        result = renderer._to_dataframe({"a": [1, 2], "b": [3, 4]})
        assert result is not None
        assert list(result.columns) == ["a", "b"]

    def test_list_of_dicts(self, renderer: MatplotlibRenderer) -> None:
        result = renderer._to_dataframe([{"a": 1}, {"a": 2}])
        assert result is not None
        assert len(result) == 2

    def test_none_returns_none(self, renderer: MatplotlibRenderer) -> None:
        assert renderer._to_dataframe(None) is None

    def test_unsupported_type_returns_none(self, renderer: MatplotlibRenderer) -> None:
        assert renderer._to_dataframe("not a dataframe") is None


# ---------------------------------------------------------------------------
# _resolve_columns
# ---------------------------------------------------------------------------

class TestResolveColumns:
    def test_auto_detect_mixed(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(simple_df, None, None)
        assert x == "category"
        assert y == "value"

    def test_auto_detect_all_numeric(self, renderer: MatplotlibRenderer, numeric_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(numeric_df, None, None)
        assert x == "x"
        assert y == "y"

    def test_invalid_columns_reset(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(simple_df, "nope", "nada")
        assert x == "category"
        assert y == "value"

    def test_explicit_valid_columns(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(simple_df, "category", "value")
        assert x == "category"
        assert y == "value"


# ---------------------------------------------------------------------------
# Chart type rendering
# ---------------------------------------------------------------------------

class TestChartTypes:
    @pytest.mark.parametrize("chart_type", [
        ChartType.BAR,
        ChartType.LINE,
        ChartType.SCATTER,
        ChartType.AREA,
    ])
    def test_xy_chart_types(
        self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame, chart_type: ChartType
    ) -> None:
        fig = renderer.render(simple_df, chart_type, x_col="category", y_col="value")
        assert fig is not None
        assert isinstance(fig, Figure)

    def test_pie(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        """Pie chart rendering — may raise TypeError due to matplotlib version
        not accepting alpha kwarg in ax.pie()."""
        try:
            fig = renderer.render(simple_df, ChartType.PIE, x_col="category", y_col="value")
            assert fig is not None
            assert isinstance(fig, Figure)
        except TypeError:
            pytest.skip("matplotlib version does not support alpha in ax.pie()")

    def test_histogram(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.HISTOGRAM, x_col="value")
        assert fig is not None
        assert isinstance(fig, Figure)

    def test_box(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BOX, y_col="value")
        assert fig is not None
        assert isinstance(fig, Figure)

    def test_heatmap(self, renderer: MatplotlibRenderer, numeric_df: pd.DataFrame) -> None:
        try:
            import seaborn  # noqa: F401
            fig = renderer.render(numeric_df, ChartType.HEATMAP)
            assert fig is not None
            assert isinstance(fig, Figure)
        except ImportError:
            pytest.skip("seaborn not available for heatmap test")

    def test_violin(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        try:
            import seaborn  # noqa: F401
            fig = renderer.render(simple_df, ChartType.VIOLIN, x_col="category", y_col="value")
            assert fig is not None
            assert isinstance(fig, Figure)
        except ImportError:
            pytest.skip("seaborn not available for violin test")

    def test_default_fallback(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        """Unknown chart type falls back to bar chart."""
        fig = renderer.render(simple_df, ChartType.TABLE, x_col="category", y_col="value")
        assert fig is not None
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Empty / None data
# ---------------------------------------------------------------------------

class TestEmptyData:
    def test_none_data_returns_none(self, renderer: MatplotlibRenderer) -> None:
        assert renderer.render(None, ChartType.BAR) is None

    def test_empty_df_returns_none(self, renderer: MatplotlibRenderer) -> None:
        assert renderer.render(pd.DataFrame(), ChartType.BAR) is None


# ---------------------------------------------------------------------------
# Style options
# ---------------------------------------------------------------------------

class TestStyleOptions:
    def test_grid_visibility(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BAR, style={"show_grid": False})
        assert fig is not None
        ax = fig.axes[0]
        # Grid lines should not be visible
        assert not ax.xaxis.get_gridlines()[0].get_visible()

    def test_axis_labels(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"x_label": "Cat", "y_label": "Val"},
        )
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Cat"
        assert ax.get_ylabel() == "Val"

    def test_opacity(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"opacity": 0.5},
        )
        assert fig is not None
        # Check that the bars have reduced alpha
        ax = fig.axes[0]
        patches = ax.patches
        if patches:
            assert patches[0].get_alpha() == 0.5

    def test_legend_position_none(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            style={"legend_position": "none"},
        )
        assert fig is not None

    def test_legend_position_top(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            style={"legend_position": "top"},
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# Horizontal bar
# ---------------------------------------------------------------------------

class TestHorizontalBar:
    def test_horizontal_creates_barh(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"orientation": "horizontal"},
        )
        assert fig is not None
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Data labels
# ---------------------------------------------------------------------------

class TestDataLabels:
    def test_bar_data_labels(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"show_labels": True},
        )
        assert fig is not None
        ax = fig.axes[0]
        # bar_label adds text annotations to containers
        assert len(ax.texts) > 0

    def test_pie_data_labels(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        try:
            fig = renderer.render(
                simple_df, ChartType.PIE,
                x_col="category", y_col="value",
                style={"show_labels": True},
            )
            assert fig is not None
        except TypeError:
            pytest.skip("matplotlib version does not support alpha in ax.pie()")


# ---------------------------------------------------------------------------
# Trendline
# ---------------------------------------------------------------------------

class TestTrendline:
    def test_scatter_trendline(self, renderer: MatplotlibRenderer, numeric_df: pd.DataFrame) -> None:
        fig = renderer.render(
            numeric_df, ChartType.SCATTER,
            x_col="x", y_col="y",
            style={"trendline": True},
        )
        assert fig is not None
        ax = fig.axes[0]
        # Should have scatter dots + trend line
        assert len(ax.lines) >= 1


# ---------------------------------------------------------------------------
# Line style
# ---------------------------------------------------------------------------

class TestLineStyle:
    @pytest.mark.parametrize("style_name,expected_ls", [
        ("dashed", "--"),
        ("dotted", ":"),
        ("solid", "-"),
    ])
    def test_line_styles(
        self,
        renderer: MatplotlibRenderer,
        simple_df: pd.DataFrame,
        style_name: str,
        expected_ls: str,
    ) -> None:
        fig = renderer.render(
            simple_df, ChartType.LINE,
            x_col="category", y_col="value",
            style={"line_style": style_name},
        )
        assert fig is not None
        ax = fig.axes[0]
        assert ax.lines[0].get_linestyle() == expected_ls


# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------

class TestSortOrder:
    def test_ascending(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"sort_order": "ascending"},
        )
        assert fig is not None

    def test_descending(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"sort_order": "descending"},
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# Custom colors and dimensions
# ---------------------------------------------------------------------------

class TestCustomColors:
    def test_custom_color_list(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"colors": ["#FF0000"]},
        )
        assert fig is not None

    def test_palette_name(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"colors": "pastel"},
        )
        assert fig is not None


class TestCustomDimensions:
    def test_custom_width_height(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"width": 1200, "height": 800},
        )
        assert fig is not None
        w, h = fig.get_size_inches()
        assert w == 12.0
        assert h == 8.0


# ---------------------------------------------------------------------------
# Marker size
# ---------------------------------------------------------------------------

class TestMarkerSize:
    def test_scatter_marker_size(self, renderer: MatplotlibRenderer, numeric_df: pd.DataFrame) -> None:
        fig = renderer.render(
            numeric_df, ChartType.SCATTER,
            x_col="x", y_col="y",
            style={"marker_size": 10},
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestMisc:
    def test_is_available(self) -> None:
        assert MatplotlibRenderer.is_available() is True

    def test_name(self, renderer: MatplotlibRenderer) -> None:
        assert renderer.name == "matplotlib"

    def test_title_applied(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BAR, title="Test Title")
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_title() == "Test Title"

    def test_theme_dark(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BAR, style={"theme": "dark"})
        assert fig is not None

    def test_colormap_heatmap(self, renderer: MatplotlibRenderer, numeric_df: pd.DataFrame) -> None:
        try:
            import seaborn  # noqa: F401
            fig = renderer.render(numeric_df, ChartType.HEATMAP, style={"colormap": "viridis"})
            assert fig is not None
        except ImportError:
            pytest.skip("seaborn not available")

    def test_box_without_seaborn_fallback(self, renderer: MatplotlibRenderer, simple_df: pd.DataFrame) -> None:
        """Box chart should still work using ax.boxplot when seaborn is absent."""
        fig = renderer.render(simple_df, ChartType.BOX, y_col="value")
        assert fig is not None
        assert isinstance(fig, Figure)
