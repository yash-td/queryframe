"""Tests for QueryResult."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from queryframe.core.result import QueryResult


@pytest.fixture
def base_result() -> QueryResult:
    """A basic QueryResult without chart."""
    return QueryResult(
        data=42,
        query="how many rows?",
        code="result = df.shape[0]",
        explanation="Row count",
        provider="mock",
        latency_ms=10.0,
    )


@pytest.fixture
def chart_result() -> QueryResult:
    """A QueryResult with a mock chart."""
    chart = MagicMock()
    chart.show = MagicMock()
    chart.to_html = MagicMock(return_value="<div>chart</div>")
    chart.write_image = MagicMock()
    return QueryResult(
        data=pd.DataFrame({"a": [1, 2]}),
        query="plot sales",
        code="result = df",
        explanation="Sales chart",
        provider="mock",
        latency_ms=15.0,
        chart=chart,
        chart_type="bar",
    )


@pytest.fixture
def engine_result() -> QueryResult:
    """A QueryResult with engine and df references for chaining."""
    engine = MagicMock()
    df = pd.DataFrame({"x": [1, 2, 3]})
    follow_up = QueryResult(
        data=3,
        query="count",
        code="result = len(df)",
        explanation="count",
        provider="mock",
        latency_ms=5.0,
    )
    engine.ask = MagicMock(return_value=follow_up)
    return QueryResult(
        data=df,
        query="show data",
        code="result = df",
        explanation="data",
        provider="mock",
        latency_ms=10.0,
        _engine=engine,
        _df=df,
    )


class TestShow:
    def test_show_with_chart_calls_chart_show(self, chart_result: QueryResult) -> None:
        chart_result.show()
        chart_result.chart.show.assert_called_once()

    def test_show_without_chart_prints_data(
        self, base_result: QueryResult, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base_result.show()
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_show_matplotlib_fallback(self) -> None:
        """When chart.show() raises AttributeError, falls back to plt.show()."""
        chart = MagicMock()
        chart.show = MagicMock(side_effect=AttributeError)
        result = QueryResult(
            data=None,
            query="plot",
            code="",
            explanation="",
            provider="mock",
            latency_ms=0,
            chart=chart,
        )
        with patch("matplotlib.pyplot.show") as mock_plt_show:
            result.show()
            mock_plt_show.assert_called_once()


class TestToHtml:
    def test_to_html_with_chart(self, chart_result: QueryResult) -> None:
        html = chart_result.to_html()
        assert html == "<div>chart</div>"
        chart_result.chart.to_html.assert_called_once()

    def test_to_html_without_chart_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1]})
        result = QueryResult(
            data=df,
            query="q",
            code="",
            explanation="",
            provider="mock",
            latency_ms=0,
        )
        html = result.to_html()
        assert "<table" in html

    def test_to_html_without_chart_plain_data(self, base_result: QueryResult) -> None:
        html = base_result.to_html()
        assert html == "42"

    def test_to_html_chart_without_to_html_method(self) -> None:
        """Chart exists but has no to_html, falls back to data."""
        chart = MagicMock(spec=[])  # no methods
        df = pd.DataFrame({"a": [1]})
        result = QueryResult(
            data=df,
            query="q",
            code="",
            explanation="",
            provider="mock",
            latency_ms=0,
            chart=chart,
        )
        html = result.to_html()
        assert "<table" in html


class TestSave:
    def test_save_chart_html(self, chart_result: QueryResult, tmp_path: object) -> None:
        path = str(tmp_path / "out.html")  # type: ignore[operator]
        chart_result.save(path)
        with open(path) as f:
            assert f.read() == "<div>chart</div>"

    def test_save_chart_image(self, chart_result: QueryResult, tmp_path: object) -> None:
        path = str(tmp_path / "out.png")  # type: ignore[operator]
        chart_result.save(path)
        chart_result.chart.write_image.assert_called_once_with(path)

    def test_save_chart_matplotlib_fallback(self, tmp_path: object) -> None:
        chart = MagicMock()
        chart.write_image = MagicMock(side_effect=AttributeError)
        chart.savefig = MagicMock()
        result = QueryResult(
            data=None,
            query="q",
            code="",
            explanation="",
            provider="mock",
            latency_ms=0,
            chart=chart,
        )
        path = str(tmp_path / "out.png")  # type: ignore[operator]
        result.save(path)
        chart.savefig.assert_called_once_with(path, bbox_inches="tight", dpi=150)

    def test_save_chart_default_extension(
        self, chart_result: QueryResult, tmp_path: object
    ) -> None:
        """Non-image, non-html extension saves as html."""
        path = str(tmp_path / "out.txt")  # type: ignore[operator]
        chart_result.save(path)
        with open(path) as f:
            assert f.read() == "<div>chart</div>"

    def test_save_dataframe_csv(self, tmp_path: object) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = QueryResult(
            data=df,
            query="q",
            code="",
            explanation="",
            provider="mock",
            latency_ms=0,
        )
        path = str(tmp_path / "out.csv")  # type: ignore[operator]
        result.save(path)
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_plain_data(self, base_result: QueryResult, tmp_path: object) -> None:
        path = str(tmp_path / "out.txt")  # type: ignore[operator]
        base_result.save(path)
        with open(path) as f:
            assert f.read() == "42"

    def test_save_returns_self(self, base_result: QueryResult, tmp_path: object) -> None:
        path = str(tmp_path / "out.txt")  # type: ignore[operator]
        returned = base_result.save(path)
        assert returned is base_result


class TestViz:
    def test_viz_with_engine(self, engine_result: QueryResult) -> None:
        new_result = engine_result.viz("matplotlib")
        engine_result._engine.ask.assert_called_once_with(
            engine_result._df,
            engine_result.query,
            viz="matplotlib",
            _previous_code=engine_result.code,
            _previous_chart_type=engine_result.chart_type,
        )
        assert new_result is not engine_result

    def test_viz_without_engine(self, base_result: QueryResult) -> None:
        returned = base_result.viz()
        assert returned is base_result


class TestAsk:
    def test_ask_chaining(self, engine_result: QueryResult) -> None:
        follow_up = engine_result.ask("count rows")
        engine_result._engine.ask.assert_called_once_with(
            engine_result._df, "count rows"
        )
        assert follow_up.data == 3

    def test_ask_without_engine_raises(self, base_result: QueryResult) -> None:
        with pytest.raises(RuntimeError, match="engine reference not available"):
            base_result.ask("follow up")


class TestRepr:
    def test_repr_basic(self, base_result: QueryResult) -> None:
        r = repr(base_result)
        assert "how many rows?" in r
        assert "latency=10ms" in r
        assert "chart=" not in r
        assert "cached=" not in r

    def test_repr_with_chart_type(self, chart_result: QueryResult) -> None:
        r = repr(chart_result)
        assert "chart='bar'" in r

    def test_repr_cached(self) -> None:
        result = QueryResult(
            data=1,
            query="q",
            code="",
            explanation="",
            provider="mock",
            latency_ms=5.0,
            cached=True,
        )
        r = repr(result)
        assert "cached=True" in r
