"""Altair declarative chart renderer with NL style support."""

from __future__ import annotations

from typing import Any

import pandas as pd

from queryframe.viz.chart_types import ChartType
from queryframe.viz.style import ChartStyle
from queryframe.viz.theme import get_theme


class AltairRenderer:
    """Renders charts using Altair's grammar-of-graphics API with NL-driven styling."""

    @property
    def name(self) -> str:
        return "altair"

    @staticmethod
    def is_available() -> bool:
        try:
            import altair  # noqa: F401
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
        import altair as alt

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return None

        cs = ChartStyle.from_dict(style)
        x_col, y_col = self._resolve_columns(df, x_col, y_col)

        # Sort data if requested
        if cs.sort_order and y_col and y_col in df.columns:
            ascending = cs.sort_order == "ascending"
            df = df.sort_values(by=y_col, ascending=ascending)

        chart = self._create_chart(alt, df, chart_type, x_col, y_col, cs)

        if chart is not None:
            if title:
                chart = chart.properties(title=title)

            width = cs.width or 600
            height = cs.height or 400
            chart = chart.properties(width=width, height=height)

            # Apply styling via configure
            chart = self._apply_style(chart, alt, cs)

        return chart

    def _create_chart(
        self,
        alt: Any,
        df: pd.DataFrame,
        chart_type: ChartType,
        x_col: str | None,
        y_col: str | None,
        cs: ChartStyle,
    ) -> Any:
        base = alt.Chart(df)
        opacity = cs.opacity or 1.0

        # Sort encoding
        x_sort = None
        if cs.sort_order and chart_type == ChartType.BAR:
            x_sort = alt.SortField(y_col, order="descending" if cs.sort_order == "descending" else "ascending") if y_col else None

        match chart_type:
            case ChartType.BAR:
                if cs.orientation == "horizontal":
                    return base.mark_bar(opacity=opacity).encode(
                        y=alt.Y(x_col, sort=x_sort) if x_col else alt.Y(),
                        x=alt.X(y_col) if y_col else alt.X(),
                    )
                return base.mark_bar(opacity=opacity).encode(
                    x=alt.X(x_col, sort=x_sort) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.LINE:
                dash = [5, 5] if cs.line_style == "dashed" else ([2, 2] if cs.line_style == "dotted" else [])
                return base.mark_line(opacity=opacity, strokeDash=dash if dash else alt.Undefined).encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.SCATTER:
                size = cs.marker_size * 10 if cs.marker_size else 60
                chart = base.mark_circle(size=size, opacity=opacity).encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
                if cs.trendline and x_col and y_col:
                    trend = chart.transform_regression(x_col, y_col).mark_line(
                        color="red", strokeDash=[5, 5]
                    )
                    chart = chart + trend
                return chart
            case ChartType.PIE:
                return base.mark_arc(opacity=opacity).encode(
                    theta=alt.Theta(y_col) if y_col else alt.Theta(),
                    color=alt.Color(x_col) if x_col else alt.Color(),
                )
            case ChartType.HISTOGRAM:
                col = x_col or y_col or str(df.columns[0])
                return base.mark_bar(opacity=opacity).encode(
                    x=alt.X(col, bin=True),
                    y="count()",
                )
            case ChartType.BOX:
                return base.mark_boxplot().encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.AREA:
                return base.mark_area(opacity=cs.opacity or 0.5).encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.HEATMAP:
                numeric = df.select_dtypes(include="number")
                if len(numeric.columns) >= 2:
                    corr = numeric.corr().reset_index().melt(id_vars="index")
                    scale = alt.Scale(scheme=cs.colormap) if cs.colormap else alt.Scale()
                    return alt.Chart(corr).mark_rect().encode(
                        x="index:N",
                        y="variable:N",
                        color=alt.Color("value:Q", scale=scale),
                    )
                return None
            case _:
                return base.mark_bar(opacity=opacity).encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )

    def _apply_style(self, chart: Any, alt: Any, cs: ChartStyle) -> Any:
        """Apply ChartStyle to an Altair chart via configure methods."""
        theme = get_theme(cs.theme)

        # Base configuration
        chart = chart.configure(
            background=theme.background,
        ).configure_axis(
            labelColor=theme.text_color,
            titleColor=theme.text_color,
            gridColor=theme.grid_color,
        ).configure_title(
            color=theme.text_color,
            fontSize=theme.title_size,
        )

        # Colors
        if cs.colors:
            chart = chart.configure_range(
                category={"scheme": cs.colors} if len(cs.colors) > 1 else None,
            )

        # Grid
        if cs.show_grid is False:
            chart = chart.configure_axis(grid=False)

        # Legend
        if cs.legend_position:
            if cs.legend_position == "none":
                chart = chart.configure_legend(disable=True)
            else:
                orient_map = {"top": "top", "bottom": "bottom", "left": "left", "right": "right"}
                chart = chart.configure_legend(orient=orient_map.get(cs.legend_position, "right"))

        return chart

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
        self, df: pd.DataFrame, x_col: str | None, y_col: str | None,
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
