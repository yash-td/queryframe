"""AST-based code validation for sandboxed execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from queryframe.sandbox.restricted import (
    ALLOWED_IMPORTS,
    FORBIDDEN_ATTRIBUTES,
    FORBIDDEN_CALLS,
)


@dataclass(frozen=True)
class ValidationResult:
    """Result of code validation."""

    is_safe: bool
    violations: tuple[str, ...]


class CodeValidator(ast.NodeVisitor):
    """Validates Python AST for dangerous operations."""

    def __init__(self) -> None:
        self._violations: list[str] = []

    def validate(self, code: str) -> ValidationResult:
        """Validate a code string and return the result."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                is_safe=False,
                violations=(f"Syntax error: {e}",),
            )

        self._violations = []
        self.visit(tree)

        return ValidationResult(
            is_safe=len(self._violations) == 0,
            violations=tuple(self._violations),
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                self._violations.append(
                    f"Forbidden import: '{alias.name}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_IMPORTS))}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                self._violations.append(
                    f"Forbidden import from: '{node.module}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_IMPORTS))}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for direct forbidden function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self._violations.append(
                    f"Forbidden function call: '{node.func.id}'"
                )

        # Check for __import__ style calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_CALLS:
                self._violations.append(
                    f"Forbidden method call: '{node.func.attr}'"
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self._violations.append(
                f"Forbidden attribute access: '{node.attr}'"
            )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._violations.append("'global' statement is forbidden")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._violations.append("'nonlocal' statement is forbidden")
        self.generic_visit(node)


def validate_code(code: str) -> ValidationResult:
    """Validate code for safe execution. Convenience function."""
    return CodeValidator().validate(code)
