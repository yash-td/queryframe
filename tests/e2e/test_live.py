"""
Comprehensive end-to-end test using a realistic mock LLM provider.
Tests the full pipeline: schema → prompt → code execution → viz → caching.
"""

import os
import tempfile
import time

import pandas as pd
import pytest

import queryframe as qf
from queryframe import QueryEngine, QueryFrameConfig
from queryframe.core import accessor as _acc
from queryframe.llm.base import LLMResponse, TokenUsage
from queryframe.sandbox.executor import execute_safe


# ── Realistic mock provider ───────────────────────────────────────────────────

class RealisticMock:
    """Returns real pandas code responses keyed by query keyword."""

    RESPONSES = {
        "total sales by region": {
            "code": 'result = df.groupby("region")["sales"].sum().reset_index().rename(columns={"sales": "total_sales"})',
            "chart_type": None,
            "explanation": "Grouped by region and summed sales.",
        },
        "highest average sales": {
            "code": 'result = df.groupby("product")["sales"].mean().idxmax()',
            "chart_type": None,
            "explanation": "Product with highest mean sales.",
        },
        "how many rows": {
            "code": "result = len(df)",
            "chart_type": None,
            "explanation": "Row count.",
        },
        "average sales": {
            "code": 'result = df["sales"].mean()',
            "chart_type": None,
            "explanation": "Mean of sales column.",
        },
        "bar chart": {
            "code": 'result = df.groupby("region")["sales"].sum().reset_index()',
            "chart_type": "bar",
            "x_col": "region",
            "y_col": "sales",
            "title": "Sales by Region",
            "explanation": "Bar chart of sales by region.",
        },
        "top region": {
            "code": 'result = df.groupby("region")["quantity"].sum().idxmax()',
            "chart_type": None,
            "explanation": "Region with most units sold.",
        },
    }

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        import json
        import re

        t = time.perf_counter()
        # Match only against the actual query line, not conversation history
        query_match = re.search(r"Query:\s*(.+)", prompt, re.IGNORECASE)
        query_line = query_match.group(1).strip().lower() if query_match else prompt.lower()

        for key, resp in self.RESPONSES.items():
            if key in query_line:
                content = json.dumps(resp)
                return LLMResponse(
                    content=content,
                    usage=TokenUsage(80, 40, 120),
                    model="mock-gpt",
                    latency_ms=(time.perf_counter() - t) * 1000,
                )
        # Fallback
        content = '{"code": "result = df.shape[0]", "chart_type": null, "explanation": "Fallback."}'
        return LLMResponse(
            content=content,
            usage=TokenUsage(80, 20, 100),
            model="mock-gpt",
            latency_ms=5.0,
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def df():
    return pd.DataFrame({
        "region":   ["North", "South", "East", "West", "North", "South"],
        "product":  ["Widget", "Widget", "Gadget", "Gadget", "Gadget", "Widget"],
        "sales":    [120, 98, 150, 110, 85, 145],
        "quantity": [10, 8, 15, 11, 9, 12],
    })


@pytest.fixture
def engine():
    return QueryEngine(
        config=QueryFrameConfig(cache_enabled=True, sandbox_enabled=True),
        provider=RealisticMock(),
    )


@pytest.fixture
def engine_no_cache():
    return QueryEngine(
        config=QueryFrameConfig(cache_enabled=False, sandbox_enabled=True),
        provider=RealisticMock(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBasicAnalytics:
    def test_groupby_returns_dataframe(self, engine, df):
        r = engine.ask(df, "total sales by region")
        assert isinstance(r.data, pd.DataFrame)
        assert len(r.data) == 4
        assert "total_sales" in r.data.columns

    def test_correct_totals(self, engine, df):
        r = engine.ask(df, "total sales by region")
        totals = dict(zip(r.data["region"], r.data["total_sales"]))
        assert totals["North"] == 205   # 120 + 85
        assert totals["South"] == 243   # 98 + 145
        assert totals["East"] == 150
        assert totals["West"] == 110

    def test_scalar_result(self, engine, df):
        r = engine.ask(df, "which product has the highest average sales?")
        assert r.data in ("Gadget", "Widget")

    def test_row_count(self, engine, df):
        r = engine.ask(df, "how many rows")
        assert r.data == 6

    def test_average_sales(self, engine, df):
        r = engine.ask(df, "average sales")
        assert abs(r.data - df["sales"].mean()) < 0.01

    def test_result_has_metadata(self, engine, df):
        r = engine.ask(df, "total sales by region")
        assert r.query == "total sales by region"
        assert r.provider == "mock"
        assert isinstance(r.code, str) and len(r.code) > 0
        assert isinstance(r.explanation, str) and len(r.explanation) > 0
        assert r.latency_ms > 0


class TestCaching:
    def test_first_call_not_cached(self, engine, df):
        r = engine.ask(df, "how many rows")
        assert not r.cached

    def test_second_call_cached(self, engine, df):
        engine.ask(df, "how many rows")
        r2 = engine.ask(df, "how many rows")
        assert r2.cached

    def test_cached_result_is_correct(self, engine, df):
        r1 = engine.ask(df, "how many rows")
        r2 = engine.ask(df, "how many rows")
        assert r1.data == r2.data == 6

    def test_cached_call_is_faster(self, engine, df):
        r1 = engine.ask(df, "average sales")
        r2 = engine.ask(df, "average sales")
        assert r2.latency_ms < r1.latency_ms

    def test_clear_cache_forces_llm_call(self, engine, df):
        engine.ask(df, "how many rows")
        engine.clear_cache()
        r = engine.ask(df, "how many rows")
        assert not r.cached

    def test_different_queries_not_cached_together(self, engine, df):
        r1 = engine.ask(df, "how many rows")
        r2 = engine.ask(df, "average sales")
        assert not r2.cached  # different query


class TestDataFrameAccessor:
    def test_qf_ask_accessor(self, engine, df):
        _acc._global_engine = engine
        r = df.qf.ask("average sales")
        assert r is not None
        assert abs(r.data - df["sales"].mean()) < 0.01

    def test_qf_config(self, engine, df):
        _acc._global_engine = engine
        # Should not raise
        df.qf.config(provider="openai")


class TestVisualization:
    def test_bar_chart_detected(self, engine, df):
        r = engine.ask(df, "bar chart of sales by region")
        assert r.chart_type == "bar"

    def test_chart_object_created(self, engine, df):
        r = engine.ask(df, "bar chart of sales by region")
        assert r.chart is not None

    def test_chart_is_plotly_figure(self, engine, df):
        r = engine.ask(df, "bar chart of sales by region")
        assert "Figure" in type(r.chart).__name__

    def test_no_chart_for_analytics(self, engine, df):
        r = engine.ask(df, "how many rows")
        assert r.chart is None
        assert r.chart_type is None


class TestSandboxSecurity:
    EVIL_PAYLOADS = [
        ("import os",          "import os\nresult = os.getcwd()"),
        ("exec() call",        "exec('result = 42')"),
        ("eval() call",        "result = eval('1+1')"),
        ("open() call",        "result = open('/etc/passwd').read()"),
        ("subprocess",         "import subprocess\nresult = subprocess.run(['ls'])"),
        ("import sys",         "import sys\nresult = sys.path"),
        ("import shutil",      "import shutil\nresult = shutil.rmtree('/')"),
        ("getattr bypass",     "result = getattr(df, '__class__')"),
        ("dunder class",       "result = df.__class__"),
        ("global statement",   "global x\nx = 1\nresult = x"),
    ]

    @pytest.mark.parametrize("label,code", EVIL_PAYLOADS)
    def test_blocked(self, df, label, code):
        res = execute_safe(code, df, timeout=5)
        assert res.error is not None, f"Expected '{label}' to be blocked, but it wasn't"

    def test_original_df_not_mutated(self, df):
        original_shape = df.shape
        original_cols = list(df.columns)
        execute_safe("df.drop(columns=['region'], inplace=True)\nresult = df", df)
        assert df.shape == original_shape
        assert list(df.columns) == original_cols


class TestSandboxAllowsSafe:
    SAFE_CASES = [
        ("groupby sum",   "result = df.groupby('region')['sales'].sum().reset_index()",  pd.DataFrame),
        ("filter rows",   "result = df[df['sales'] > 100]",                              pd.DataFrame),
        ("numpy mean",    "import numpy as np\nresult = np.mean(df['sales'])",           float),
        ("string ops",    "result = df['region'].str.upper().tolist()",                  list),
        ("multi-line",    "g = df.groupby('product')['sales'].sum()\nresult = g.to_dict()", dict),
        ("math import",   "import math\nresult = math.sqrt(df['sales'].sum())",          float),
        ("datetime",      "from datetime import date\nresult = str(date.today())",       str),
    ]

    @pytest.mark.parametrize("label,code,expected_type", SAFE_CASES)
    def test_allowed(self, df, label, code, expected_type):
        res = execute_safe(code, df, timeout=5)
        assert res.error is None, f"'{label}' should be allowed but got: {res.error}"
        assert isinstance(res.data, expected_type), f"Expected {expected_type}, got {type(res.data)}"


class TestSandboxTimeout:
    def test_infinite_loop_killed(self, df):
        from queryframe.utils.errors import SandboxTimeoutError
        with pytest.raises(SandboxTimeoutError):
            execute_safe("while True: pass", df, timeout=1, skip_validation=True)


class TestSchemaExtraction:
    def test_shape_correct(self, df):
        from queryframe.core.schema import extract_schema
        s = extract_schema(df)
        assert s.shape == (6, 4)

    def test_all_columns_present(self, df):
        from queryframe.core.schema import extract_schema
        s = extract_schema(df)
        names = [c.name for c in s.columns]
        assert "region" in names and "sales" in names

    def test_fingerprint_stable(self, df):
        from queryframe.core.schema import extract_schema
        fp1 = extract_schema(df).fingerprint
        fp2 = extract_schema(df).fingerprint
        assert fp1 == fp2

    def test_fingerprint_changes_with_data(self, df):
        from queryframe.core.schema import extract_schema
        fp1 = extract_schema(df).fingerprint
        df2 = df.drop(columns=["sales"])
        fp2 = extract_schema(df2).fingerprint
        assert fp1 != fp2

    def test_compression_contains_key_info(self, df):
        from queryframe.core.schema import extract_schema, compress_schema
        s = extract_schema(df)
        c = compress_schema(s, "sales by region")
        assert "region" in c and "sales" in c and "6" in c


class TestConversationMemory:
    def test_memory_grows_with_queries(self, engine_no_cache, df):
        engine_no_cache.ask(df, "total sales by region")
        engine_no_cache.ask(df, "how many rows")
        assert len(engine_no_cache._conversation) == 2

    def test_clear_memory(self, engine_no_cache, df):
        engine_no_cache.ask(df, "total sales by region")
        engine_no_cache.clear_memory()
        assert len(engine_no_cache._conversation) == 0


class TestQueryResultAPI:
    def test_repr(self, engine, df):
        r = engine.ask(df, "total sales by region")
        assert "QueryResult" in repr(r)
        assert "total sales by region" in repr(r)

    def test_save_csv(self, engine, df):
        r = engine.ask(df, "total sales by region")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            r.save(path)
            assert os.path.exists(path)
            loaded = pd.read_csv(path)
            assert len(loaded) == 4

    def test_save_html_chart(self, engine, df):
        r = engine.ask(df, "bar chart of sales by region")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chart.html")
            r.save(path)
            assert os.path.exists(path)
            content = open(path).read()
            assert "<html" in content.lower() or "plotly" in content.lower()

    def test_chaining_ask(self, engine, df):
        r = engine.ask(df, "how many rows")
        assert r.data == 6
        # .ask() chains to a follow-up query on the same DataFrame
        r2 = r.ask("average sales")
        assert abs(r2.data - df["sales"].mean()) < 0.01

    def test_to_html_returns_string(self, engine, df):
        r = engine.ask(df, "bar chart of sales by region")
        html = r.to_html()
        assert isinstance(html, str) and len(html) > 0
