"""Tests for ChartStyle, get_theme, and COLOR_PALETTES."""

from __future__ import annotations

import pytest

from queryframe.viz.style import COLOR_PALETTES, ChartStyle
from queryframe.viz.theme import (
    DEFAULT_THEME,
    DARK_THEME,
    MINIMAL_THEME,
    PRESENTATION_THEME,
    QueryFrameTheme,
    get_theme,
)


# ---------------------------------------------------------------------------
# ChartStyle.from_dict
# ---------------------------------------------------------------------------

class TestFromDictNoneAndEmpty:
    def test_none_returns_defaults(self) -> None:
        cs = ChartStyle.from_dict(None)
        assert cs.is_empty

    def test_empty_dict_returns_defaults(self) -> None:
        cs = ChartStyle.from_dict({})
        assert cs.is_empty

    def test_non_dict_returns_defaults(self) -> None:
        cs = ChartStyle.from_dict("not a dict")  # type: ignore[arg-type]
        assert cs.is_empty


class TestFromDictFullValid:
    def test_full_valid_dict(self) -> None:
        raw = {
            "theme": "Dark",
            "orientation": "Horizontal",
            "sort_order": "Descending",
            "line_style": "Dashed",
            "colormap": "viridis",
            "legend_position": "Top",
            "show_grid": True,
            "show_labels": False,
            "trendline": True,
            "marker_size": 10,
            "width": 800,
            "height": 600,
            "opacity": 0.7,
            "x_label": "X Axis",
            "y_label": "Y Axis",
            "colors": ["#FF0000", "#00FF00"],
        }
        cs = ChartStyle.from_dict(raw)

        # String fields lowercased (except labels)
        assert cs.theme == "dark"
        assert cs.orientation == "horizontal"
        assert cs.sort_order == "descending"
        assert cs.line_style == "dashed"
        assert cs.colormap == "viridis"
        assert cs.legend_position == "top"

        # Labels preserve case
        assert cs.x_label == "X Axis"
        assert cs.y_label == "Y Axis"

        # Bool fields
        assert cs.show_grid is True
        assert cs.show_labels is False
        assert cs.trendline is True

        # Int fields
        assert cs.marker_size == 10
        assert cs.width == 800
        assert cs.height == 600

        # Float fields
        assert cs.opacity == 0.7

        # Colors
        assert cs.colors == ["#FF0000", "#00FF00"]

    def test_is_empty_false_when_set(self) -> None:
        cs = ChartStyle.from_dict({"theme": "dark"})
        assert not cs.is_empty


class TestFromDictUnknownKeys:
    def test_unknown_keys_ignored(self) -> None:
        cs = ChartStyle.from_dict({"unknown_key": "value", "another": 42})
        assert cs.is_empty

    def test_mixed_known_unknown(self) -> None:
        cs = ChartStyle.from_dict({"theme": "dark", "unknown": True})
        assert cs.theme == "dark"
        assert cs.is_empty is False


class TestFromDictStringFields:
    def test_string_lowercased(self) -> None:
        cs = ChartStyle.from_dict({"theme": "DARK", "orientation": "HORIZONTAL"})
        assert cs.theme == "dark"
        assert cs.orientation == "horizontal"

    def test_labels_not_lowercased(self) -> None:
        cs = ChartStyle.from_dict({"x_label": "Revenue ($M)", "y_label": "Quarter"})
        assert cs.x_label == "Revenue ($M)"
        assert cs.y_label == "Quarter"

    def test_whitespace_stripped(self) -> None:
        cs = ChartStyle.from_dict({"theme": "  dark  "})
        assert cs.theme == "dark"

    def test_empty_string_ignored(self) -> None:
        cs = ChartStyle.from_dict({"theme": "", "orientation": "   "})
        assert cs.theme is None
        assert cs.orientation is None


class TestFromDictBoolFields:
    def test_bool_true(self) -> None:
        cs = ChartStyle.from_dict({"show_grid": True, "show_labels": True, "trendline": True})
        assert cs.show_grid is True
        assert cs.show_labels is True
        assert cs.trendline is True

    def test_bool_false(self) -> None:
        cs = ChartStyle.from_dict({"show_grid": False})
        assert cs.show_grid is False

    def test_non_bool_ignored(self) -> None:
        cs = ChartStyle.from_dict({"show_grid": "yes", "show_labels": 1})
        assert cs.show_grid is None
        assert cs.show_labels is None


class TestFromDictIntFields:
    def test_positive_int(self) -> None:
        cs = ChartStyle.from_dict({"marker_size": 8, "width": 1000, "height": 500})
        assert cs.marker_size == 8
        assert cs.width == 1000
        assert cs.height == 500

    def test_float_converted_to_int(self) -> None:
        cs = ChartStyle.from_dict({"marker_size": 7.5})
        assert cs.marker_size == 7

    def test_negative_rejected(self) -> None:
        cs = ChartStyle.from_dict({"marker_size": -5, "width": -100, "height": 0})
        assert cs.marker_size is None
        assert cs.width is None
        assert cs.height is None

    def test_zero_rejected(self) -> None:
        cs = ChartStyle.from_dict({"marker_size": 0})
        assert cs.marker_size is None


class TestFromDictOpacity:
    def test_valid_opacity(self) -> None:
        cs = ChartStyle.from_dict({"opacity": 0.5})
        assert cs.opacity == 0.5

    def test_opacity_zero(self) -> None:
        cs = ChartStyle.from_dict({"opacity": 0})
        assert cs.opacity == 0.0

    def test_opacity_one(self) -> None:
        cs = ChartStyle.from_dict({"opacity": 1})
        assert cs.opacity == 1.0

    def test_opacity_out_of_range_high(self) -> None:
        cs = ChartStyle.from_dict({"opacity": 1.5})
        assert cs.opacity is None

    def test_opacity_out_of_range_low(self) -> None:
        cs = ChartStyle.from_dict({"opacity": -0.1})
        assert cs.opacity is None


class TestFromDictColors:
    def test_list_of_colors(self) -> None:
        cs = ChartStyle.from_dict({"colors": ["#FF0000", "#00FF00", "#0000FF"]})
        assert cs.colors == ["#FF0000", "#00FF00", "#0000FF"]

    def test_palette_name(self) -> None:
        cs = ChartStyle.from_dict({"colors": "pastel"})
        assert cs.colors == COLOR_PALETTES["pastel"]

    def test_palette_name_case_insensitive(self) -> None:
        cs = ChartStyle.from_dict({"colors": "Pastel"})
        assert cs.colors == COLOR_PALETTES["pastel"]

    def test_single_color_string(self) -> None:
        cs = ChartStyle.from_dict({"colors": "#FF0000"})
        assert cs.colors == ["#FF0000"]

    def test_unknown_palette_becomes_single_color(self) -> None:
        cs = ChartStyle.from_dict({"colors": "not_a_palette"})
        assert cs.colors == ["not_a_palette"]

    def test_empty_values_filtered(self) -> None:
        cs = ChartStyle.from_dict({"colors": ["#FF0000", "", None, "#00FF00"]})
        assert cs.colors == ["#FF0000", "#00FF00"]


# ---------------------------------------------------------------------------
# is_empty property
# ---------------------------------------------------------------------------

class TestIsEmpty:
    def test_default_is_empty(self) -> None:
        assert ChartStyle().is_empty

    def test_any_field_set_not_empty(self) -> None:
        assert not ChartStyle(theme="dark").is_empty
        assert not ChartStyle(opacity=0.5).is_empty
        assert not ChartStyle(show_grid=True).is_empty
        assert not ChartStyle(colors=["red"]).is_empty


# ---------------------------------------------------------------------------
# get_theme
# ---------------------------------------------------------------------------

class TestGetTheme:
    def test_dark(self) -> None:
        theme = get_theme("dark")
        assert theme is DARK_THEME

    def test_minimal(self) -> None:
        theme = get_theme("minimal")
        assert theme is MINIMAL_THEME

    def test_presentation(self) -> None:
        theme = get_theme("presentation")
        assert theme is PRESENTATION_THEME

    def test_none_returns_default(self) -> None:
        theme = get_theme(None)
        assert theme is DEFAULT_THEME

    def test_unknown_returns_default(self) -> None:
        theme = get_theme("nonexistent_theme")
        assert theme is DEFAULT_THEME

    def test_case_insensitive(self) -> None:
        assert get_theme("DARK") is DARK_THEME
        assert get_theme("Dark") is DARK_THEME

    def test_whitespace_stripped(self) -> None:
        assert get_theme("  dark  ") is DARK_THEME

    def test_light_alias(self) -> None:
        assert get_theme("light") is DEFAULT_THEME


# ---------------------------------------------------------------------------
# QueryFrameTheme output methods
# ---------------------------------------------------------------------------

class TestQueryFrameTheme:
    def test_plotly_template_structure(self) -> None:
        tmpl = DEFAULT_THEME.plotly_template()
        assert "layout" in tmpl
        assert "colorway" in tmpl["layout"]
        assert "paper_bgcolor" in tmpl["layout"]
        assert "font" in tmpl["layout"]

    def test_matplotlib_rcparams_structure(self) -> None:
        rc = DEFAULT_THEME.matplotlib_rcparams()
        assert "figure.facecolor" in rc
        assert "axes.facecolor" in rc
        assert "grid.color" in rc

    def test_dark_theme_background(self) -> None:
        assert DARK_THEME.background == "#2E3440"

    def test_presentation_larger_fonts(self) -> None:
        assert PRESENTATION_THEME.font_size > DEFAULT_THEME.font_size
        assert PRESENTATION_THEME.title_size > DEFAULT_THEME.title_size


# ---------------------------------------------------------------------------
# COLOR_PALETTES
# ---------------------------------------------------------------------------

class TestColorPalettes:
    @pytest.mark.parametrize("palette_name", [
        "pastel", "vibrant", "earth", "ocean", "sunset", "monochrome", "neon", "corporate",
    ])
    def test_palette_exists(self, palette_name: str) -> None:
        assert palette_name in COLOR_PALETTES

    def test_palettes_are_non_empty_lists(self) -> None:
        for name, colors in COLOR_PALETTES.items():
            assert isinstance(colors, list), f"{name} is not a list"
            assert len(colors) > 0, f"{name} is empty"

    def test_palette_values_are_strings(self) -> None:
        for name, colors in COLOR_PALETTES.items():
            for c in colors:
                assert isinstance(c, str), f"{name} contains non-string color: {c}"
