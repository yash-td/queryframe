"""Tests for PlotlyRenderer — chart creation, data conversion, column resolution, and styling."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from queryframe.viz.chart_types import ChartType
from queryframe.viz.style import ChartStyle

plotly = pytest.importorskip("plotly")
go = pytest.importorskip("plotly.graph_objects")

from queryframe.viz.plotly_renderer import PlotlyRenderer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def renderer() -> PlotlyRenderer:
    return PlotlyRenderer()


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["A", "B", "C", "D"],
        "value": [10, 20, 30, 40],
    })


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
        "z": [5, 3, 1, 7, 9],
    })


# ---------------------------------------------------------------------------
# _to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    def test_dataframe_passthrough(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        result = renderer._to_dataframe(simple_df)
        assert result is not None
        pd.testing.assert_frame_equal(result, simple_df)

    def test_series(self, renderer: PlotlyRenderer) -> None:
        s = pd.Series([10, 20, 30], name="val")
        result = renderer._to_dataframe(s)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "val" in result.columns

    def test_dict(self, renderer: PlotlyRenderer) -> None:
        result = renderer._to_dataframe({"a": [1, 2], "b": [3, 4]})
        assert result is not None
        assert list(result.columns) == ["a", "b"]

    def test_list_of_dicts(self, renderer: PlotlyRenderer) -> None:
        result = renderer._to_dataframe([{"a": 1}, {"a": 2}])
        assert result is not None
        assert len(result) == 2

    def test_none_returns_none(self, renderer: PlotlyRenderer) -> None:
        assert renderer._to_dataframe(None) is None

    def test_unsupported_type_returns_none(self, renderer: PlotlyRenderer) -> None:
        assert renderer._to_dataframe("not a dataframe") is None


# ---------------------------------------------------------------------------
# _resolve_columns
# ---------------------------------------------------------------------------

class TestResolveColumns:
    def test_auto_detect_mixed_columns(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(simple_df, None, None, ChartType.BAR)
        assert x == "category"
        assert y == "value"

    def test_auto_detect_all_numeric(self, renderer: PlotlyRenderer, numeric_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(numeric_df, None, None, ChartType.SCATTER)
        assert x == "x"
        assert y == "y"

    def test_invalid_columns_reset_to_none(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(simple_df, "nonexistent", "also_missing", ChartType.BAR)
        # Should fall back to auto-detection
        assert x == "category"
        assert y == "value"

    def test_explicit_valid_columns_kept(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        x, y = renderer._resolve_columns(simple_df, "category", "value", ChartType.BAR)
        assert x == "category"
        assert y == "value"


# ---------------------------------------------------------------------------
# Chart type rendering — every supported type should produce a Figure
# ---------------------------------------------------------------------------

class TestChartTypes:
    @pytest.mark.parametrize("chart_type", [
        ChartType.BAR,
        ChartType.LINE,
        ChartType.SCATTER,
        ChartType.AREA,
        ChartType.FUNNEL,
    ])
    def test_xy_chart_types(
        self, renderer: PlotlyRenderer, simple_df: pd.DataFrame, chart_type: ChartType
    ) -> None:
        fig = renderer.render(simple_df, chart_type, x_col="category", y_col="value")
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_pie(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.PIE, x_col="category", y_col="value")
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_histogram(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.HISTOGRAM, x_col="value")
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_heatmap(self, renderer: PlotlyRenderer, numeric_df: pd.DataFrame) -> None:
        fig = renderer.render(numeric_df, ChartType.HEATMAP)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_box(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BOX, y_col="value")
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_violin(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.VIOLIN, y_col="value")
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_treemap(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.TREEMAP, x_col="category", y_col="value")
        assert fig is not None
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Empty / None data
# ---------------------------------------------------------------------------

class TestEmptyData:
    def test_none_data_returns_none(self, renderer: PlotlyRenderer) -> None:
        assert renderer.render(None, ChartType.BAR) is None

    def test_empty_df_returns_none(self, renderer: PlotlyRenderer) -> None:
        empty = pd.DataFrame()
        assert renderer.render(empty, ChartType.BAR) is None

    def test_heatmap_no_numeric_returns_none(self, renderer: PlotlyRenderer) -> None:
        df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        assert renderer.render(df, ChartType.HEATMAP) is None


# ---------------------------------------------------------------------------
# _apply_style
# ---------------------------------------------------------------------------

class TestApplyStyle:
    def test_grid_visibility(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BAR, style={"show_grid": False})
        assert fig is not None
        # Grid setting is applied — xaxis and yaxis showgrid should be False
        assert fig.layout.xaxis.showgrid is False
        assert fig.layout.yaxis.showgrid is False

    def test_opacity(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BAR, style={"opacity": 0.5})
        assert fig is not None
        assert fig.data[0].opacity == 0.5

    def test_axis_labels(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"x_label": "My X", "y_label": "My Y"},
        )
        assert fig is not None
        assert fig.layout.xaxis.title.text == "My X"
        assert fig.layout.yaxis.title.text == "My Y"

    def test_marker_size_scatter(self, renderer: PlotlyRenderer, numeric_df: pd.DataFrame) -> None:
        fig = renderer.render(
            numeric_df, ChartType.SCATTER,
            x_col="x", y_col="y",
            style={"marker_size": 15},
        )
        assert fig is not None
        assert fig.data[0].marker.size == 15

    def test_line_dash_style(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.LINE,
            x_col="category", y_col="value",
            style={"line_style": "dashed"},
        )
        assert fig is not None
        assert fig.data[0].line.dash == "dash"

    def test_data_labels_bar(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"show_labels": True},
        )
        assert fig is not None
        assert fig.data[0].texttemplate is not None

    def test_data_labels_pie(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.PIE,
            x_col="category", y_col="value",
            style={"show_labels": True},
        )
        assert fig is not None
        assert "percent" in (fig.data[0].textinfo or "")

    def test_legend_position_none(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            style={"legend_position": "none"},
        )
        assert fig is not None
        assert fig.layout.showlegend is False

    def test_legend_position_top(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            style={"legend_position": "top"},
        )
        assert fig is not None
        assert fig.layout.legend.orientation == "h"


# ---------------------------------------------------------------------------
# Horizontal bar
# ---------------------------------------------------------------------------

class TestHorizontalBar:
    def test_horizontal_orientation(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"orientation": "horizontal"},
        )
        assert fig is not None
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------

class TestSortOrder:
    def test_ascending(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"sort_order": "ascending"},
        )
        assert fig is not None

    def test_descending(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
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
    def test_custom_color_list(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"colors": ["#FF0000", "#00FF00"]},
        )
        assert fig is not None

    def test_single_color_bar(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"colors": ["#FF0000"]},
        )
        assert fig is not None

    def test_palette_name(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.PIE,
            x_col="category", y_col="value",
            style={"colors": "pastel"},
        )
        assert fig is not None


class TestCustomDimensions:
    def test_custom_width_height(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(
            simple_df, ChartType.BAR,
            x_col="category", y_col="value",
            style={"width": 800, "height": 400},
        )
        assert fig is not None
        assert fig.layout.width == 800
        assert fig.layout.height == 400


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestMisc:
    def test_is_available(self) -> None:
        assert PlotlyRenderer.is_available() is True

    def test_name(self, renderer: PlotlyRenderer) -> None:
        assert renderer.name == "plotly"

    def test_title_applied(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BAR, title="Test Title")
        assert fig is not None
        assert fig.layout.title.text == "Test Title"

    def test_theme_dark(self, renderer: PlotlyRenderer, simple_df: pd.DataFrame) -> None:
        fig = renderer.render(simple_df, ChartType.BAR, style={"theme": "dark"})
        assert fig is not None

    def test_colormap_heatmap(self, renderer: PlotlyRenderer, numeric_df: pd.DataFrame) -> None:
        fig = renderer.render(numeric_df, ChartType.HEATMAP, style={"colormap": "viridis"})
        assert fig is not None
