"""Prompt construction and LLM response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jinja2 import Template

from queryframe.llm.prompt.templates import (
    ANALYSIS_TEMPLATE,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_LOCAL,
    VISUALIZATION_TEMPLATE,
)


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in the conversation history."""

    query: str
    summary: str


@dataclass(frozen=True)
class ParsedResponse:
    """Parsed response from the LLM."""

    code: str
    chart_type: str | None
    x_col: str | None
    y_col: str | None
    title: str | None
    explanation: str
    style: dict[str, Any] | None = None


# Keywords that suggest a visualization request
_VIZ_KEYWORDS = frozenset({
    "chart", "plot", "graph", "visualize", "visualise", "show", "draw",
    "bar", "line", "scatter", "pie", "histogram", "heatmap", "box", "area",
    "trend", "distribution", "comparison",
})


def is_viz_query(query: str) -> bool:
    """Detect if the query is asking for a visualization."""
    words = set(query.lower().split())
    return bool(words & _VIZ_KEYWORDS)


def build_prompt(
    query: str,
    schema_str: str,
    conversation_history: list[ConversationTurn] | None = None,
    is_local: bool = False,
) -> tuple[str, str]:
    """Build the system prompt and user prompt.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    template_str = VISUALIZATION_TEMPLATE if is_viz_query(query) else ANALYSIS_TEMPLATE

    template = Template(template_str)
    user_prompt = template.render(
        schema=schema_str,
        query=query,
        conversation_history=conversation_history or [],
    )

    system = SYSTEM_PROMPT_LOCAL if is_local else SYSTEM_PROMPT
    return system, user_prompt


def parse_llm_response(raw: str) -> ParsedResponse:
    """Parse the LLM response into a structured result.

    Handles multiple formats:
    1. Valid JSON
    2. JSON wrapped in markdown code fences
    3. Plain code without JSON wrapper
    """
    # Try to extract JSON from code fences
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1).strip()

    # Try direct JSON parse
    try:
        data = json.loads(raw)
        return _from_dict(data)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group())
            return _from_dict(data)
        except json.JSONDecodeError:
            pass

    # Fallback: treat the whole response as code
    code = raw.strip()
    # Remove any markdown code fences
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)

    return ParsedResponse(
        code=code,
        chart_type=None,
        x_col=None,
        y_col=None,
        title=None,
        explanation="Generated code from LLM response.",
    )


def _from_dict(data: dict[str, Any]) -> ParsedResponse:
    """Convert a dict to a ParsedResponse."""
    return ParsedResponse(
        code=data.get("code", ""),
        chart_type=data.get("chart_type"),
        x_col=data.get("x_col"),
        y_col=data.get("y_col"),
        title=data.get("title"),
        explanation=data.get("explanation", ""),
        style=data.get("style"),
    )
