"""Unified theming across visualization libraries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryFrameTheme:
    """Theme configuration for consistent chart styling."""

    # Color palette (modern, accessible)
    colors: tuple[str, ...] = (
        "#636EFA",  # blue
        "#EF553B",  # red
        "#00CC96",  # green
        "#AB63FA",  # purple
        "#FFA15A",  # orange
        "#19D3F3",  # cyan
        "#FF6692",  # pink
        "#B6E880",  # lime
        "#FF97FF",  # magenta
        "#FECB52",  # yellow
    )
    background: str = "#FFFFFF"
    text_color: str = "#2E3440"
    grid_color: str = "#E5E9F0"
    font_family: str = "Inter, -apple-system, sans-serif"
    font_size: int = 12
    title_size: int = 16

    def plotly_template(self) -> dict:
        """Generate a Plotly template dict from this theme."""
        return {
            "layout": {
                "colorway": list(self.colors),
                "paper_bgcolor": self.background,
                "plot_bgcolor": self.background,
                "font": {
                    "family": self.font_family,
                    "size": self.font_size,
                    "color": self.text_color,
                },
                "title": {"font": {"size": self.title_size}},
                "xaxis": {
                    "gridcolor": self.grid_color,
                    "zerolinecolor": self.grid_color,
                },
                "yaxis": {
                    "gridcolor": self.grid_color,
                    "zerolinecolor": self.grid_color,
                },
            }
        }

    def matplotlib_rcparams(self) -> dict:
        """Generate matplotlib rcParams from this theme."""
        return {
            "figure.facecolor": self.background,
            "axes.facecolor": self.background,
            "axes.edgecolor": self.grid_color,
            "axes.grid": True,
            "grid.color": self.grid_color,
            "grid.alpha": 0.5,
            "text.color": self.text_color,
            "font.family": "sans-serif",
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.prop_cycle": f"cycler('color', {list(self.colors)})",
        }


# Default theme instance
DEFAULT_THEME = QueryFrameTheme()

# Dark mode theme
DARK_THEME = QueryFrameTheme(
    background="#2E3440",
    text_color="#ECEFF4",
    grid_color="#4C566A",
)

# Minimal theme — clean, whitespace-heavy
MINIMAL_THEME = QueryFrameTheme(
    colors=(
        "#4A90D9", "#E74C3C", "#27AE60", "#8E44AD",
        "#F39C12", "#16A085", "#D35400", "#2980B9",
        "#C0392B", "#1ABC9C",
    ),
    background="#FFFFFF",
    text_color="#555555",
    grid_color="#F0F0F0",
    font_size=11,
    title_size=14,
)

# Presentation theme — bold, high contrast, larger fonts
PRESENTATION_THEME = QueryFrameTheme(
    colors=(
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
        "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F1948A", "#82E0AA",
    ),
    background="#1A1A2E",
    text_color="#EAEAEA",
    grid_color="#2D2D44",
    font_family="Helvetica, Arial, sans-serif",
    font_size=16,
    title_size=24,
)

# Theme registry
_THEMES: dict[str, QueryFrameTheme] = {
    "default": DEFAULT_THEME,
    "dark": DARK_THEME,
    "minimal": MINIMAL_THEME,
    "presentation": PRESENTATION_THEME,
    "light": DEFAULT_THEME,
}


def get_theme(name: str | None) -> QueryFrameTheme:
    """Look up a theme by name. Returns DEFAULT_THEME for unknown names."""
    if name is None:
        return DEFAULT_THEME
    return _THEMES.get(name.lower().strip(), DEFAULT_THEME)
