"""Restricted builtins and allowed imports for sandboxed execution."""

from __future__ import annotations

# Only these builtins are available in the sandbox
RESTRICTED_BUILTINS: dict[str, object] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

# Only these modules can be imported in sandbox code
ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "pandas",
    "pd",
    "numpy",
    "np",
    "math",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "re",
    "statistics",
})

# These attribute names are forbidden (dunder access)
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
    "__import__",
    "__builtins__",
    "__globals__",
    "__code__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
    "__dict__",
    "__module__",
    "__reduce__",
    "__reduce_ex__",
    "__setattr__",
    "__delattr__",
})

# These function names are forbidden
FORBIDDEN_CALLS: frozenset[str] = frozenset({
    "exec",
    "eval",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "exit",
    "quit",
    "input",
    "help",
})
