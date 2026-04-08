"""Plotly interactive chart renderer with NL style support."""

from __future__ import annotations

from typing import Any

import pandas as pd

from queryframe.viz.chart_types import ChartType
from queryframe.viz.style import ChartStyle
from queryframe.viz.theme import DEFAULT_THEME, get_theme


class PlotlyRenderer:
    """Renders charts using Plotly Express with NL-driven styling."""

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
        style: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        import plotly.express as px
        import plotly.graph_objects as go

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return None

        cs = ChartStyle.from_dict(style)
        x_col, y_col = self._resolve_columns(df, x_col, y_col, chart_type)

        # Sort data if requested
        if cs.sort_order and y_col and y_col in df.columns:
            ascending = cs.sort_order == "ascending"
            df = df.sort_values(by=y_col, ascending=ascending)

        # Select theme
        theme = get_theme(cs.theme)
        template = theme.plotly_template()

        # Override colors if specified
        if cs.colors:
            template["layout"]["colorway"] = cs.colors

        fig = self._create_figure(px, df, chart_type, x_col, y_col, title, cs, **kwargs)

        if fig is not None:
            fig.update_layout(
                template=go.layout.Template(layout=go.Layout(**template["layout"])),
                title=title,
                margin=dict(l=60, r=30, t=50, b=50),
            )
            self._apply_style(fig, cs, chart_type)

            # Custom dimensions
            if cs.width or cs.height:
                fig.update_layout(
                    width=cs.width or fig.layout.width,
                    height=cs.height or fig.layout.height,
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
        cs: ChartStyle,
        **kwargs: Any,
    ) -> Any:
        common: dict[str, Any] = {"title": title}

        # Handle horizontal orientation for bar charts
        if cs.orientation == "horizontal" and chart_type == ChartType.BAR:
            common["orientation"] = "h"
            x_col, y_col = y_col, x_col

        # Handle trendline for scatter
        scatter_kwargs: dict[str, Any] = {}
        if cs.trendline and chart_type == ChartType.SCATTER:
            try:
                import statsmodels  # noqa: F401
                scatter_kwargs["trendline"] = "ols"
            except ImportError:
                pass  # trendline requires statsmodels

        # Handle colormap for heatmaps
        colorscale = cs.colormap if cs.colormap else None

        match chart_type:
            case ChartType.BAR:
                color_arg = {}
                if cs.colors and len(cs.colors) == 1:
                    color_arg["color_discrete_sequence"] = cs.colors
                return px.bar(df, x=x_col, y=y_col, **common, **color_arg, **kwargs)
            case ChartType.LINE:
                return px.line(df, x=x_col, y=y_col, **common, **kwargs)
            case ChartType.SCATTER:
                return px.scatter(df, x=x_col, y=y_col, **common, **scatter_kwargs, **kwargs)
            case ChartType.PIE:
                names_col = x_col or df.columns[0]
                values_col = y_col or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                pie_kwargs: dict[str, Any] = {}
                if cs.colors:
                    pie_kwargs["color_discrete_sequence"] = cs.colors
                return px.pie(df, names=names_col, values=values_col, **common, **pie_kwargs, **kwargs)
            case ChartType.HISTOGRAM:
                col = x_col or y_col or df.columns[0]
                return px.histogram(df, x=col, **common, **kwargs)
            case ChartType.HEATMAP:
                numeric_df = df.select_dtypes(include="number")
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    return px.imshow(
                        corr, text_auto=True,
                        color_continuous_scale=colorscale,
                        **common,
                    )
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

    def _apply_style(self, fig: Any, cs: ChartStyle, chart_type: ChartType) -> None:
        """Apply ChartStyle options to a Plotly figure."""
        # Grid
        if cs.show_grid is not None:
            fig.update_xaxes(showgrid=cs.show_grid)
            fig.update_yaxes(showgrid=cs.show_grid)

        # Opacity
        if cs.opacity is not None:
            fig.update_traces(opacity=cs.opacity)

        # Axis labels
        if cs.x_label:
            fig.update_xaxes(title_text=cs.x_label)
        if cs.y_label:
            fig.update_yaxes(title_text=cs.y_label)

        # Marker size (scatter)
        if cs.marker_size and chart_type == ChartType.SCATTER:
            fig.update_traces(marker_size=cs.marker_size)

        # Line style
        if cs.line_style and chart_type == ChartType.LINE:
            dash_map = {"dashed": "dash", "dotted": "dot", "solid": "solid"}
            fig.update_traces(line_dash=dash_map.get(cs.line_style, cs.line_style))

        # Data labels
        if cs.show_labels:
            if chart_type == ChartType.PIE:
                fig.update_traces(textinfo="percent+label")
            elif chart_type == ChartType.BAR:
                fig.update_traces(texttemplate="%{value:.2s}", textposition="outside")

        # Legend position
        if cs.legend_position:
            legend_map = {
                "top":    dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                "bottom": dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                "left":   dict(x=-0.15, y=0.5),
                "right":  dict(x=1.05, y=0.5),
                "none":   dict(visible=False),
            }
            legend_opts = legend_map.get(cs.legend_position)
            if legend_opts:
                visible = legend_opts.pop("visible", True)
                fig.update_layout(showlegend=visible, legend=legend_opts if visible else {})

    def _to_dataframe(self, data: Any) -> pd.DataFrame | None:
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
        if x_col and x_col not in df.columns:
            x_col = None
        if y_col and y_col not in df.columns:
            y_col = None

        if x_col is None and y_col is None and len(df.columns) >= 2:
            non_numeric = df.select_dtypes(exclude="number").columns
            numeric = df.select_dtypes(include="number").columns
            if len(non_numeric) > 0 and len(numeric) > 0:
                x_col = str(non_numeric[0])
                y_col = str(numeric[0])
            else:
                x_col = str(df.columns[0])
                y_col = str(df.columns[1])

        return x_col, y_col
