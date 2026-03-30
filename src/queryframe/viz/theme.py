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
