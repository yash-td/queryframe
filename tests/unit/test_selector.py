"""Tests for chart type resolution and inference."""

import pytest

from queryframe.viz.chart_types import ChartType, infer_chart_type, resolve_chart_type


class TestResolveChartType:
    def test_exact_match(self):
        assert resolve_chart_type("bar") == ChartType.BAR
        assert resolve_chart_type("line") == ChartType.LINE
        assert resolve_chart_type("scatter") == ChartType.SCATTER

    def test_aliases(self):
        assert resolve_chart_type("barchart") == ChartType.BAR
        assert resolve_chart_type("scatter plot") == ChartType.SCATTER
        assert resolve_chart_type("hist") == ChartType.HISTOGRAM

    def test_case_insensitive(self):
        assert resolve_chart_type("BAR") == ChartType.BAR
        assert resolve_chart_type("Line") == ChartType.LINE

    def test_none_input(self):
        assert resolve_chart_type(None) is None

    def test_unknown(self):
        assert resolve_chart_type("unknown_type") is None


class TestInferChartType:
    def test_trend(self):
        assert infer_chart_type("show the sales trend over time") == ChartType.LINE

    def test_distribution(self):
        assert infer_chart_type("show the distribution of prices") == ChartType.HISTOGRAM

    def test_correlation(self):
        assert infer_chart_type("scatter plot of price vs quantity") == ChartType.SCATTER

    def test_proportion(self):
        assert infer_chart_type("show the percentage breakdown") == ChartType.PIE

    def test_default_bar(self):
        assert infer_chart_type("show sales by region") == ChartType.BAR
