"""Matplotlib/Seaborn static chart renderer with NL style support."""

from __future__ import annotations

from typing import Any

import pandas as pd

from queryframe.viz.chart_types import ChartType
from queryframe.viz.style import ChartStyle
from queryframe.viz.theme import DEFAULT_THEME, get_theme


class MatplotlibRenderer:
    """Renders charts using Matplotlib and Seaborn with NL-driven styling."""

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
        style: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return None

        cs = ChartStyle.from_dict(style)
        x_col, y_col = self._resolve_columns(df, x_col, y_col)

        # Sort data if requested
        if cs.sort_order and y_col and y_col in df.columns:
            ascending = cs.sort_order == "ascending"
            df = df.sort_values(by=y_col, ascending=ascending)

        # Apply theme
        theme = get_theme(cs.theme)
        rc = theme.matplotlib_rcparams()
        for key, val in rc.items():
            if key != "axes.prop_cycle":
                plt.rcParams[key] = val

        width = (cs.width or 1000) / 100
        height = (cs.height or 600) / 100
        fig, ax = plt.subplots(figsize=(width, height))

        # Get colors
        colors = cs.colors or list(theme.colors)

        self._draw(ax, df, chart_type, x_col, y_col, cs, colors, **kwargs)

        if title:
            ax.set_title(title, fontsize=theme.title_size, pad=15)

        # Apply style options
        if cs.show_grid is not None:
            ax.grid(cs.show_grid)
        if cs.x_label:
            ax.set_xlabel(cs.x_label)
        if cs.y_label:
            ax.set_ylabel(cs.y_label)
        if cs.legend_position:
            if cs.legend_position == "none":
                ax.legend().set_visible(False) if ax.get_legend() else None
            else:
                loc_map = {"top": "upper center", "bottom": "lower center",
                           "left": "center left", "right": "center right"}
                ax.legend(loc=loc_map.get(cs.legend_position, "best"))

        fig.tight_layout()
        return fig

    def _draw(
        self,
        ax: Any,
        df: pd.DataFrame,
        chart_type: ChartType,
        x_col: str | None,
        y_col: str | None,
        cs: ChartStyle,
        colors: list[str],
        **kwargs: Any,
    ) -> None:
        alpha = cs.opacity if cs.opacity is not None else 1.0

        try:
            import seaborn as sns
            has_seaborn = True
        except ImportError:
            has_seaborn = False

        match chart_type:
            case ChartType.BAR:
                if x_col and y_col:
                    if cs.orientation == "horizontal":
                        ax.barh(df[x_col].astype(str), df[y_col], color=colors[0], alpha=alpha)
                        ax.set_ylabel(x_col)
                        ax.set_xlabel(y_col)
                    else:
                        bars = ax.bar(df[x_col].astype(str), df[y_col], color=colors[0], alpha=alpha)
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.tick_params(axis="x", rotation=45)
                    if cs.show_labels:
                        if cs.orientation == "horizontal":
                            ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)
                        else:
                            ax.bar_label(bars, fmt="%.0f", padding=3)
            case ChartType.LINE:
                if x_col and y_col:
                    linestyle = {"dashed": "--", "dotted": ":", "solid": "-"}.get(
                        cs.line_style or "solid", "-"
                    )
                    ax.plot(df[x_col], df[y_col], color=colors[0], linewidth=2,
                            linestyle=linestyle, alpha=alpha)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            case ChartType.SCATTER:
                if x_col and y_col:
                    size = cs.marker_size ** 2 if cs.marker_size else 50
                    ax.scatter(df[x_col], df[y_col], color=colors[0], alpha=alpha or 0.7, s=size)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    if cs.trendline:
                        import numpy as np
                        x_num = pd.to_numeric(df[x_col], errors="coerce").dropna()
                        y_num = pd.to_numeric(df[y_col], errors="coerce").dropna()
                        idx = x_num.index.intersection(y_num.index)
                        if len(idx) > 1:
                            z = np.polyfit(x_num[idx], y_num[idx], 1)
                            p = np.poly1d(z)
                            ax.plot(x_num[idx].sort_values(), p(x_num[idx].sort_values()),
                                    color=colors[1] if len(colors) > 1 else "red",
                                    linestyle="--", linewidth=1.5, label="Trend")
                            ax.legend()
            case ChartType.PIE:
                col = y_col or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                labels = df[x_col].astype(str) if x_col else df.index.astype(str)
                pie_colors = colors[:len(df)]
                autopct = "%1.1f%%" if cs.show_labels else None
                ax.pie(df[col], labels=labels, colors=pie_colors, autopct=autopct, alpha=alpha)
            case ChartType.HISTOGRAM:
                col = x_col or y_col or df.columns[0]
                ax.hist(df[col].dropna(), bins=30, color=colors[0], edgecolor="white", alpha=alpha)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
            case ChartType.HEATMAP:
                if has_seaborn:
                    numeric_df = df.select_dtypes(include="number")
                    cmap = cs.colormap or "coolwarm"
                    sns.heatmap(numeric_df.corr(), annot=True, cmap=cmap, ax=ax)
            case ChartType.BOX:
                if has_seaborn and y_col:
                    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                elif y_col:
                    ax.boxplot(df[y_col].dropna())
                    ax.set_ylabel(y_col)
            case ChartType.AREA:
                if x_col and y_col:
                    ax.fill_between(df[x_col], df[y_col], alpha=cs.opacity or 0.4, color=colors[0])
                    ax.plot(df[x_col], df[y_col], color=colors[0], linewidth=2)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            case ChartType.VIOLIN:
                if has_seaborn and y_col:
                    sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
            case _:
                if x_col and y_col:
                    ax.bar(df[x_col].astype(str), df[y_col], color=colors[0], alpha=alpha)

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
