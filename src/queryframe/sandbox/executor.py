"""Safe code execution engine."""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from queryframe.sandbox.restricted import ALLOWED_IMPORTS, RESTRICTED_BUILTINS
from queryframe.sandbox.timeout import execution_timeout
from queryframe.sandbox.validator import validate_code
from queryframe.utils.errors import SandboxError, SandboxValidationError


@dataclass(frozen=True)
class ExecutionResult:
    """Result of sandboxed code execution."""

    data: Any
    stdout: str
    error: str | None
    execution_ms: float
    variables: dict[str, Any]


def execute_safe(
    code: str,
    df: pd.DataFrame,
    timeout: int = 30,
    skip_validation: bool = False,
) -> ExecutionResult:
    """Execute code in a restricted sandbox.

    The code receives:
    - `df`: a copy of the DataFrame
    - `pd`: pandas module
    - `np`: numpy module
    - Restricted builtins (no exec, eval, open, etc.)

    The code must store its result in a variable called `result`.
    """
    start = time.perf_counter()

    # Step 1: Validate the code
    if not skip_validation:
        validation = validate_code(code)
        if not validation.is_safe:
            return ExecutionResult(
                data=None,
                stdout="",
                error=f"Code validation failed: {'; '.join(validation.violations)}",
                execution_ms=(time.perf_counter() - start) * 1000,
                variables={},
            )

    # Step 2: Build restricted namespace with safe import function
    _SAFE_MODULES = {
        "pandas": pd,
        "pd": pd,
        "numpy": np,
        "np": np,
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "time": __import__("time"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
        "operator": __import__("operator"),
        "re": __import__("re"),
        "statistics": __import__("statistics"),
        "decimal": __import__("decimal"),
        "fractions": __import__("fractions"),
        "random": __import__("random"),
        "string": __import__("string"),
    }

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")
        if root in _SAFE_MODULES:
            return _SAFE_MODULES[root]
        raise ImportError(f"Module '{name}' is not available in sandbox")

    builtins = {**RESTRICTED_BUILTINS, "__import__": _safe_import}

    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }

    # Step 3: Execute with timeout and stdout capture
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = stdout_capture
        with execution_timeout(timeout):
            exec(code, namespace)  # noqa: S102 — controlled sandbox

        # Extract result
        result_data = namespace.get("result")

        # Collect chart-related variables
        variables = {}
        for key in ("result", "chart_type", "x_col", "y_col", "title"):
            if key in namespace:
                variables[key] = namespace[key]

        return ExecutionResult(
            data=result_data,
            stdout=stdout_capture.getvalue(),
            error=None,
            execution_ms=(time.perf_counter() - start) * 1000,
            variables=variables,
        )

    except SandboxError:
        raise
    except Exception as e:
        return ExecutionResult(
            data=None,
            stdout=stdout_capture.getvalue(),
            error=f"{type(e).__name__}: {e}",
            execution_ms=(time.perf_counter() - start) * 1000,
            variables={},
        )
    finally:
        sys.stdout = old_stdout
