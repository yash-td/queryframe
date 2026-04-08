"""Tests for AST-based code validation — SECURITY CRITICAL."""

import pytest

from queryframe.sandbox.validator import validate_code


class TestValidatorRejectsUnsafe:
    """These tests MUST all pass — they verify the sandbox catches dangerous code."""

    def test_reject_import_os(self):
        result = validate_code("import os")
        assert not result.is_safe
        assert any("os" in v for v in result.violations)

    def test_reject_import_subprocess(self):
        result = validate_code("import subprocess")
        assert not result.is_safe

    def test_reject_from_os_import(self):
        result = validate_code("from os import system")
        assert not result.is_safe

    def test_reject_import_sys(self):
        result = validate_code("import sys")
        assert not result.is_safe

    def test_reject_import_shutil(self):
        result = validate_code("import shutil")
        assert not result.is_safe

    def test_reject_exec(self):
        result = validate_code("exec('print(1)')")
        assert not result.is_safe
        assert any("exec" in v for v in result.violations)

    def test_reject_eval(self):
        result = validate_code("eval('1+1')")
        assert not result.is_safe

    def test_reject_open(self):
        result = validate_code("open('/etc/passwd')")
        assert not result.is_safe

    def test_reject_dunder_import(self):
        result = validate_code("__import__('os')")
        assert not result.is_safe

    def test_reject_compile(self):
        result = validate_code("compile('code', 'f', 'exec')")
        assert not result.is_safe

    def test_reject_globals(self):
        result = validate_code("globals()")
        assert not result.is_safe

    def test_reject_getattr(self):
        result = validate_code("getattr(df, '__class__')")
        assert not result.is_safe

    def test_reject_dunder_builtins(self):
        result = validate_code("x = df.__builtins__")
        assert not result.is_safe

    def test_reject_dunder_class(self):
        result = validate_code("x = df.__class__")
        assert not result.is_safe

    def test_reject_dunder_subclasses(self):
        result = validate_code("x = str.__subclasses__()")
        assert not result.is_safe

    def test_reject_global_statement(self):
        result = validate_code("global x")
        assert not result.is_safe

    def test_reject_nonlocal_statement(self):
        result = validate_code("def f():\n    nonlocal x")
        assert not result.is_safe

    def test_reject_breakpoint(self):
        result = validate_code("breakpoint()")
        assert not result.is_safe

    def test_reject_input(self):
        result = validate_code("input('enter password')")
        assert not result.is_safe

    def test_reject_vars(self):
        result = validate_code("result = vars()")
        assert not result.is_safe

    def test_reject_dir(self):
        result = validate_code("result = dir(df)")
        assert not result.is_safe

    def test_reject_setattr_call(self):
        result = validate_code("setattr(df, 'x', 1)")
        assert not result.is_safe

    def test_reject_delattr_call(self):
        result = validate_code("delattr(df, 'x')")
        assert not result.is_safe

    def test_reject_dunder_reduce(self):
        result = validate_code("result = df.__reduce__()")
        assert not result.is_safe

    def test_reject_dunder_globals(self):
        result = validate_code("result = df.__globals__")
        assert not result.is_safe

    def test_reject_dunder_code(self):
        result = validate_code("result = (lambda: 0).__code__")
        assert not result.is_safe

    def test_reject_star_import_os(self):
        result = validate_code("from os import *")
        assert not result.is_safe

    def test_reject_async_function(self):
        result = validate_code("async def f(): pass")
        assert not result.is_safe

    def test_reject_await(self):
        result = validate_code("async def f(): await something()")
        assert not result.is_safe

    def test_reject_dunder_file(self):
        result = validate_code("result = pd.__file__")
        assert not result.is_safe

    def test_reject_dunder_name(self):
        result = validate_code("result = pd.__name__")
        assert not result.is_safe

    def test_reject_subscript_dunder(self):
        """Block dict-style dunder access like obj['__builtins__']."""
        result = validate_code("result = x['__builtins__']")
        assert not result.is_safe

    def test_reject_subscript_dunder_import(self):
        result = validate_code("result = x['__import__']")
        assert not result.is_safe

    def test_reject_fstring_import(self):
        """f-string with __import__ call."""
        result = validate_code("result = f'{__import__(\"os\")}'")
        assert not result.is_safe

    def test_reject_object_subclasses_chain(self):
        """Block object traversal attacks."""
        result = validate_code("result = ().__class__.__bases__[0].__subclasses__()")
        assert not result.is_safe

    def test_reject_type_subclasses(self):
        result = validate_code("result = type.__subclasses__(type)")
        assert not result.is_safe


class TestValidatorAllowsSafe:
    """These tests verify that normal pandas/numpy code is allowed."""

    def test_allow_pandas_operations(self):
        result = validate_code("result = df['sales'].sum()")
        assert result.is_safe

    def test_allow_groupby(self):
        result = validate_code("result = df.groupby('region')['sales'].mean()")
        assert result.is_safe

    def test_allow_filtering(self):
        result = validate_code("result = df[df['sales'] > 100]")
        assert result.is_safe

    def test_allow_import_pandas(self):
        result = validate_code("import pandas as pd")
        assert result.is_safe

    def test_allow_import_numpy(self):
        result = validate_code("import numpy as np")
        assert result.is_safe

    def test_allow_import_math(self):
        result = validate_code("import math")
        assert result.is_safe

    def test_allow_import_datetime(self):
        result = validate_code("from datetime import datetime")
        assert result.is_safe

    def test_allow_list_comprehension(self):
        result = validate_code("result = [x * 2 for x in range(10)]")
        assert result.is_safe

    def test_allow_lambda(self):
        result = validate_code("result = df.apply(lambda x: x * 2)")
        assert result.is_safe

    def test_allow_string_operations(self):
        result = validate_code("result = df['name'].str.upper()")
        assert result.is_safe

    def test_allow_merge(self):
        code = "result = pd.merge(df, df, on='region')"
        result = validate_code(code)
        assert result.is_safe

    def test_allow_multi_line(self):
        code = """
grouped = df.groupby('region')['sales'].sum()
result = grouped.reset_index()
"""
        result = validate_code(code)
        assert result.is_safe

    def test_allow_pandas_query_method(self):
        result = validate_code("result = df.query('sales > 100')")
        assert result.is_safe


class TestValidatorEdgeCases:
    def test_syntax_error(self):
        result = validate_code("def f(:")
        assert not result.is_safe
        assert any("Syntax error" in v for v in result.violations)

    def test_empty_code(self):
        result = validate_code("")
        assert result.is_safe

    def test_multiple_violations(self):
        result = validate_code("import os\nexec('code')")
        assert not result.is_safe
        assert len(result.violations) >= 2
