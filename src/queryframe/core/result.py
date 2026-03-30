"""QueryResult — unified result type for all queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryResult:
    """Immutable result from a QueryFrame query."""

    data: Any
    query: str
    code: str
    explanation: str
    provider: str
    latency_ms: float
    chart: Any = None
    chart_type: str | None = None
    cached: bool = False
    _engine: Any = field(default=None, repr=False, compare=False)
    _df: Any = field(default=None, repr=False, compare=False)

    def show(self) -> None:
        """Display the chart if available, otherwise print the data."""
        if self.chart is not None:
            try:
                self.chart.show()
            except AttributeError:
                # matplotlib figure
                import matplotlib.pyplot as plt

                plt.show()
        else:
            print(self.data)

    def to_html(self) -> str:
        """Convert the chart to HTML string."""
        if self.chart is not None:
            try:
                return self.chart.to_html()
            except AttributeError:
                pass
        if hasattr(self.data, "to_html"):
            return self.data.to_html()
        return str(self.data)

    def save(self, path: str) -> QueryResult:
        """Save the chart or data to a file."""
        if self.chart is not None:
            if path.endswith(".html"):
                with open(path, "w") as f:
                    f.write(self.to_html())
            elif path.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
                try:
                    self.chart.write_image(path)
                except AttributeError:
                    self.chart.savefig(path, bbox_inches="tight", dpi=150)
            else:
                with open(path, "w") as f:
                    f.write(self.to_html())
        elif hasattr(self.data, "to_csv"):
            self.data.to_csv(path, index=False)
        else:
            with open(path, "w") as f:
                f.write(str(self.data))
        return self

    def viz(self, renderer: str = "plotly") -> QueryResult:
        """Re-render the result with a different visualization library."""
        if self._engine is not None and self._df is not None:
            return self._engine.ask(
                self._df,
                self.query,
                viz=renderer,
                _previous_code=self.code,
                _previous_chart_type=self.chart_type,
            )
        return self

    def ask(self, query: str) -> QueryResult:
        """Ask a follow-up question using conversation context."""
        if self._engine is not None and self._df is not None:
            return self._engine.ask(self._df, query)
        raise RuntimeError("Cannot chain queries — engine reference not available")

    def __repr__(self) -> str:
        parts = [f"QueryResult(query='{self.query}'"]
        if self.chart_type:
            parts.append(f", chart='{self.chart_type}'")
        if self.cached:
            parts.append(", cached=True")
        parts.append(f", latency={self.latency_ms:.0f}ms)")
        return "".join(parts)
