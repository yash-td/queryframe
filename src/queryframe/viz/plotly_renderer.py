"""Plotly interactive chart renderer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from queryframe.viz.chart_types import ChartType
from queryframe.viz.theme import DEFAULT_THEME


class PlotlyRenderer:
    """Renders charts using Plotly Express."""

    @property
    def name(self) -> str:
        return "plotly"

    @staticmethod
    def is_available() -> bool:
        try:
            import plotly  # noqa: F401
            return True
        except ImportError:
            return False

    def render(
        self,
        data: Any,
        chart_type: ChartType,
        x_col: str | None = None,
        y_col: str | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        import plotly.express as px
        import plotly.graph_objects as go

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return None

        x_col, y_col = self._resolve_columns(df, x_col, y_col, chart_type)
        theme = DEFAULT_THEME.plotly_template()

        fig = self._create_figure(px, df, chart_type, x_col, y_col, title, **kwargs)

        if fig is not None:
            fig.update_layout(
                template=go.layout.Template(layout=go.Layout(**theme["layout"])),
                title=title,
                margin=dict(l=60, r=30, t=50, b=50),
            )

        return fig

    def _create_figure(
        self,
        px: Any,
        df: pd.DataFrame,
        chart_type: ChartType,
        x_col: str | None,
        y_col: str | None,
        title: str | None,
        **kwargs: Any,
    ) -> Any:
        """Create a plotly figure for the given chart type."""
        common = {"title": title}

        match chart_type:
            case ChartType.BAR:
                return px.bar(df, x=x_col, y=y_col, **common, **kwargs)
            case ChartType.LINE:
                return px.line(df, x=x_col, y=y_col, **common, **kwargs)
            case ChartType.SCATTER:
                return px.scatter(df, x=x_col, y=y_col, **common, **kwargs)
            case ChartType.PIE:
                names_col = x_col or df.columns[0]
                values_col = y_col or df.columns[1] if len(df.columns) > 1 else df.columns[0]
                return px.pie(df, names=names_col, values=values_col, **common, **kwargs)
            case ChartType.HISTOGRAM:
                col = x_col or y_col or df.columns[0]
                return px.histogram(df, x=col, **common, **kwargs)
            case ChartType.HEATMAP:
                numeric_df = df.select_dtypes(include="number")
                if not numeric_df.empty:
                    import plotly.figure_factory as ff
                    corr = numeric_df.corr()
                    fig = px.imshow(corr, text_auto=True, **common)
                    return fig
                return None
            case ChartType.BOX:
                col = y_col or x_col or df.columns[0]
                return px.box(df, y=col, x=x_col, **common, **kwargs)
            case ChartType.AREA:
                return px.area(df, x=x_col, y=y_col, **common, **kwargs)
            case ChartType.VIOLIN:
                col = y_col or x_col or df.columns[0]
                return px.violin(df, y=col, x=x_col, **common, **kwargs)
            case ChartType.TREEMAP:
                return px.treemap(df, path=[x_col] if x_col else None, values=y_col, **common, **kwargs)
            case ChartType.FUNNEL:
                return px.funnel(df, x=x_col, y=y_col, **common, **kwargs)
            case _:
                return px.bar(df, x=x_col, y=y_col, **common, **kwargs)

    def _to_dataframe(self, data: Any) -> pd.DataFrame | None:
        """Convert various data types to a DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.reset_index()
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, (list, tuple)):
            return pd.DataFrame(data)
        return None

    def _resolve_columns(
        self,
        df: pd.DataFrame,
        x_col: str | None,
        y_col: str | None,
        chart_type: ChartType,
    ) -> tuple[str | None, str | None]:
        """Resolve x and y column names, falling back to auto-detection."""
        if x_col and x_col not in df.columns:
            x_col = None
        if y_col and y_col not in df.columns:
            y_col = None

        if x_col is None and y_col is None and len(df.columns) >= 2:
            # Auto-detect: first non-numeric = x, first numeric = y
            non_numeric = df.select_dtypes(exclude="number").columns
            numeric = df.select_dtypes(include="number").columns
            if len(non_numeric) > 0 and len(numeric) > 0:
                x_col = str(non_numeric[0])
                y_col = str(numeric[0])
            else:
                x_col = str(df.columns[0])
                y_col = str(df.columns[1])

        return x_col, y_col
