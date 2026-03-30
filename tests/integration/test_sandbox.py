"""Integration tests for sandbox execution."""

import pandas as pd
import pytest

from queryframe.sandbox.executor import execute_safe
from queryframe.utils.errors import SandboxTimeoutError


@pytest.fixture
def df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85, 92, 78],
    })


class TestSandboxExecution:
    def test_simple_computation(self, df):
        result = execute_safe("result = df['score'].mean()", df)
        assert result.error is None
        assert result.data == pytest.approx(85.0)

    def test_groupby(self, df):
        result = execute_safe(
            "result = df.groupby('name')['score'].sum().reset_index()",
            df,
        )
        assert result.error is None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3

    def test_filtering(self, df):
        result = execute_safe("result = df[df['score'] > 80]", df)
        assert result.error is None
        assert len(result.data) == 2

    def test_variable_extraction(self, df):
        code = """
result = df.groupby('name')['score'].sum().reset_index()
chart_type = 'bar'
x_col = 'name'
y_col = 'score'
title = 'Scores by Name'
"""
        result = execute_safe(code, df)
        assert result.error is None
        assert result.variables["chart_type"] == "bar"
        assert result.variables["x_col"] == "name"

    def test_original_df_not_modified(self, df):
        original_shape = df.shape
        execute_safe("df.drop(columns=['name'], inplace=True)\nresult = df", df)
        assert df.shape == original_shape  # original unchanged

    def test_stdout_captured(self, df):
        result = execute_safe("print('hello')\nresult = 42", df)
        assert "hello" in result.stdout
        assert result.data == 42

    def test_runtime_error_caught(self, df):
        result = execute_safe("result = 1 / 0", df)
        assert result.error is not None
        assert "ZeroDivision" in result.error

    def test_unsafe_code_blocked(self, df):
        result = execute_safe("import os\nresult = os.getcwd()", df)
        assert result.error is not None
        assert "validation failed" in result.error.lower()

    def test_timeout(self, df):
        with pytest.raises(SandboxTimeoutError):
            execute_safe(
                "while True: pass",
                df,
                timeout=1,
                skip_validation=True,
            )

    def test_numpy_available(self, df):
        result = execute_safe("import numpy as np\nresult = np.mean([1, 2, 3])", df)
        assert result.error is None
        assert result.data == pytest.approx(2.0)

    def test_execution_time_tracked(self, df):
        result = execute_safe("result = 42", df)
        assert result.execution_ms > 0
