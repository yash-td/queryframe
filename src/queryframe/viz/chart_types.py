"""Chart type taxonomy and inference."""

from __future__ import annotations

from enum import Enum


class ChartType(Enum):
    """Supported chart types."""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    AREA = "area"
    TABLE = "table"
    VIOLIN = "violin"
    TREEMAP = "treemap"
    FUNNEL = "funnel"


# Map string names to ChartType
CHART_TYPE_MAP: dict[str, ChartType] = {
    ct.value: ct for ct in ChartType
}

# Common aliases
CHART_TYPE_MAP.update({
    "barchart": ChartType.BAR,
    "bar_chart": ChartType.BAR,
    "bar chart": ChartType.BAR,
    "linechart": ChartType.LINE,
    "line_chart": ChartType.LINE,
    "line chart": ChartType.LINE,
    "scatterplot": ChartType.SCATTER,
    "scatter_plot": ChartType.SCATTER,
    "scatter plot": ChartType.SCATTER,
    "piechart": ChartType.PIE,
    "pie_chart": ChartType.PIE,
    "pie chart": ChartType.PIE,
    "hist": ChartType.HISTOGRAM,
    "boxplot": ChartType.BOX,
    "box_plot": ChartType.BOX,
    "box plot": ChartType.BOX,
    "heat_map": ChartType.HEATMAP,
    "heat map": ChartType.HEATMAP,
    "area_chart": ChartType.AREA,
    "area chart": ChartType.AREA,
    "tree_map": ChartType.TREEMAP,
    "tree map": ChartType.TREEMAP,
})


def resolve_chart_type(raw: str | None) -> ChartType | None:
    """Resolve a raw string to a ChartType enum value."""
    if raw is None:
        return None
    return CHART_TYPE_MAP.get(raw.lower().strip())


def infer_chart_type(query: str, n_rows: int = 0, n_cols: int = 0) -> ChartType:
    """Infer chart type from query text when the LLM doesn't specify one."""
    q = query.lower()

    if any(w in q for w in ("trend", "over time", "timeseries", "time series", "growth")):
        return ChartType.LINE
    if any(w in q for w in ("distribution", "histogram", "frequency")):
        return ChartType.HISTOGRAM
    if any(w in q for w in ("scatter", "correlation", "relationship", "vs", "versus")):
        return ChartType.SCATTER
    if any(w in q for w in ("proportion", "percentage", "share", "pie", "breakdown")):
        return ChartType.PIE
    if any(w in q for w in ("heatmap", "heat map", "matrix", "correlation matrix")):
        return ChartType.HEATMAP
    if any(w in q for w in ("box", "boxplot", "outlier", "quartile", "median")):
        return ChartType.BOX
    if any(w in q for w in ("area", "stacked area", "cumulative")):
        return ChartType.AREA

    # Default to bar for comparison-style queries
    return ChartType.BAR
