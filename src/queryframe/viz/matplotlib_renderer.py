"""Matplotlib/Seaborn static chart renderer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from queryframe.viz.chart_types import ChartType
from queryframe.viz.theme import DEFAULT_THEME


class MatplotlibRenderer:
    """Renders charts using Matplotlib and Seaborn."""

    @property
    def name(self) -> str:
        return "matplotlib"

    @staticmethod
    def is_available() -> bool:
        try:
            import matplotlib  # noqa: F401
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return None

        x_col, y_col = self._resolve_columns(df, x_col, y_col)

        # Apply theme
        rc = DEFAULT_THEME.matplotlib_rcparams()
        for key, val in rc.items():
            if key != "axes.prop_cycle":
                plt.rcParams[key] = val

        fig, ax = plt.subplots(figsize=(10, 6))

        self._draw(ax, df, chart_type, x_col, y_col, **kwargs)

        if title:
            ax.set_title(title, fontsize=DEFAULT_THEME.title_size, pad=15)

        fig.tight_layout()
        return fig

    def _draw(
        self,
        ax: Any,
        df: pd.DataFrame,
        chart_type: ChartType,
        x_col: str | None,
        y_col: str | None,
        **kwargs: Any,
    ) -> None:
        """Draw the chart on the given axes."""
        colors = DEFAULT_THEME.colors

        try:
            import seaborn as sns
            has_seaborn = True
        except ImportError:
            has_seaborn = False

        match chart_type:
            case ChartType.BAR:
                if x_col and y_col:
                    ax.bar(df[x_col].astype(str), df[y_col], color=colors[0])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.tick_params(axis="x", rotation=45)
            case ChartType.LINE:
                if x_col and y_col:
                    ax.plot(df[x_col], df[y_col], color=colors[0], linewidth=2)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            case ChartType.SCATTER:
                if x_col and y_col:
                    ax.scatter(df[x_col], df[y_col], color=colors[0], alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            case ChartType.PIE:
                col = y_col or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                labels = df[x_col].astype(str) if x_col else df.index.astype(str)
                ax.pie(df[col], labels=labels, colors=colors[:len(df)], autopct="%1.1f%%")
            case ChartType.HISTOGRAM:
                col = x_col or y_col or df.columns[0]
                ax.hist(df[col].dropna(), bins=30, color=colors[0], edgecolor="white")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
            case ChartType.HEATMAP:
                if has_seaborn:
                    numeric_df = df.select_dtypes(include="number")
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            case ChartType.BOX:
                if has_seaborn and y_col:
                    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                elif y_col:
                    ax.boxplot(df[y_col].dropna())
                    ax.set_ylabel(y_col)
            case ChartType.AREA:
                if x_col and y_col:
                    ax.fill_between(df[x_col], df[y_col], alpha=0.4, color=colors[0])
                    ax.plot(df[x_col], df[y_col], color=colors[0], linewidth=2)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            case ChartType.VIOLIN:
                if has_seaborn and y_col:
                    sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
            case _:
                if x_col and y_col:
                    ax.bar(df[x_col].astype(str), df[y_col], color=colors[0])

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
