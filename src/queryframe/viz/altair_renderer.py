"""Altair declarative chart renderer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from queryframe.viz.chart_types import ChartType


class AltairRenderer:
    """Renders charts using Altair's grammar-of-graphics API."""

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
        **kwargs: Any,
    ) -> Any:
        import altair as alt

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return None

        x_col, y_col = self._resolve_columns(df, x_col, y_col)

        chart = self._create_chart(alt, df, chart_type, x_col, y_col)

        if chart is not None and title:
            chart = chart.properties(title=title)

        if chart is not None:
            chart = chart.properties(width=600, height=400)

        return chart

    def _create_chart(
        self,
        alt: Any,
        df: pd.DataFrame,
        chart_type: ChartType,
        x_col: str | None,
        y_col: str | None,
    ) -> Any:
        base = alt.Chart(df)

        match chart_type:
            case ChartType.BAR:
                return base.mark_bar().encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.LINE:
                return base.mark_line().encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.SCATTER:
                return base.mark_circle(size=60).encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.PIE:
                return base.mark_arc().encode(
                    theta=alt.Theta(y_col) if y_col else alt.Theta(),
                    color=alt.Color(x_col) if x_col else alt.Color(),
                )
            case ChartType.HISTOGRAM:
                col = x_col or y_col or str(df.columns[0])
                return base.mark_bar().encode(
                    x=alt.X(col, bin=True),
                    y="count()",
                )
            case ChartType.BOX:
                return base.mark_boxplot().encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.AREA:
                return base.mark_area(opacity=0.5).encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )
            case ChartType.HEATMAP:
                numeric = df.select_dtypes(include="number")
                if len(numeric.columns) >= 2:
                    corr = numeric.corr().reset_index().melt(id_vars="index")
                    return alt.Chart(corr).mark_rect().encode(
                        x="index:N",
                        y="variable:N",
                        color="value:Q",
                    )
                return None
            case _:
                return base.mark_bar().encode(
                    x=alt.X(x_col) if x_col else alt.X(),
                    y=alt.Y(y_col) if y_col else alt.Y(),
                )

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
