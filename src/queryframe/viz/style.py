"""Chart style dataclass — single source of truth for styling options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Named color palettes users can reference in natural language
COLOR_PALETTES: dict[str, list[str]] = {
    "pastel": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#E8BAFF", "#FFD4BA"],
    "vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"],
    "earth": ["#8B7355", "#CD853F", "#DEB887", "#D2B48C", "#C4A882", "#A0522D"],
    "ocean": ["#006994", "#40E0D0", "#0077B6", "#00B4D8", "#48CAE4", "#90E0EF"],
    "sunset": ["#FF6B35", "#F7C59F", "#EFEFD0", "#004E89", "#1A659E", "#FF4365"],
    "monochrome": ["#2C3E50", "#7F8C8D", "#95A5A6", "#BDC3C7", "#D5DBDB", "#ECF0F1"],
    "neon": ["#FF00FF", "#00FFFF", "#FF6600", "#39FF14", "#FF3F8E", "#04D9FF"],
    "corporate": ["#003366", "#336699", "#6699CC", "#99CCFF", "#CCE5FF", "#E6F2FF"],
}


@dataclass(frozen=True)
class ChartStyle:
    """Immutable chart styling options parsed from natural language."""

    theme: str | None = None
    colors: list[str] | None = None
    orientation: str | None = None       # "horizontal" or "vertical"
    sort_order: str | None = None        # "ascending" or "descending"
    line_style: str | None = None        # "solid", "dashed", "dotted"
    show_grid: bool | None = None
    show_labels: bool | None = None      # data labels / percentage labels
    marker_size: int | None = None
    colormap: str | None = None          # "viridis", "coolwarm", etc.
    opacity: float | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend_position: str | None = None   # "top", "bottom", "left", "right", "none"
    trendline: bool | None = None
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ChartStyle:
        """Safely construct a ChartStyle from LLM output, ignoring unknown keys."""
        if not raw or not isinstance(raw, dict):
            return cls()

        kwargs: dict[str, Any] = {}

        # String fields
        for key in ("theme", "orientation", "sort_order", "line_style",
                     "colormap", "x_label", "y_label", "legend_position"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                kwargs[key] = val.strip().lower() if key != "x_label" and key != "y_label" else val.strip()

        # Bool fields
        for key in ("show_grid", "show_labels", "trendline"):
            val = raw.get(key)
            if isinstance(val, bool):
                kwargs[key] = val

        # Int fields
        for key in ("marker_size", "width", "height"):
            val = raw.get(key)
            if isinstance(val, (int, float)) and val > 0:
                kwargs[key] = int(val)

        # Float fields
        if "opacity" in raw:
            val = raw["opacity"]
            if isinstance(val, (int, float)) and 0 <= val <= 1:
                kwargs["opacity"] = float(val)

        # Colors — can be a list of color strings or a palette name
        colors_raw = raw.get("colors")
        if isinstance(colors_raw, list):
            kwargs["colors"] = [str(c) for c in colors_raw if c]
        elif isinstance(colors_raw, str):
            palette = COLOR_PALETTES.get(colors_raw.lower())
            if palette:
                kwargs["colors"] = palette
            else:
                kwargs["colors"] = [colors_raw]

        return cls(**kwargs)

    @property
    def is_empty(self) -> bool:
        """Check if no style options were specified."""
        return all(
            getattr(self, f.name) is None
            for f in self.__dataclass_fields__.values()
        )
