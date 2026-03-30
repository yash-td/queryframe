"""Tests for prompt building and response parsing."""

import pytest

from queryframe.llm.prompt.builder import (
    ConversationTurn,
    build_prompt,
    is_viz_query,
    parse_llm_response,
)


class TestIsVizQuery:
    def test_chart_keywords(self):
        assert is_viz_query("show me a bar chart of sales")
        assert is_viz_query("plot the trend over time")
        assert is_viz_query("visualize the distribution")
        assert is_viz_query("draw a scatter plot")

    def test_non_viz_queries(self):
        assert not is_viz_query("what is the average sales?")
        assert not is_viz_query("how many rows are there?")
        assert not is_viz_query("filter to region North")


class TestBuildPrompt:
    def test_returns_system_and_user(self):
        system, user = build_prompt("average sales", "schema here")
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert "schema here" in user
        assert "average sales" in user

    def test_viz_query_uses_viz_template(self):
        system, user = build_prompt("show me a bar chart", "schema")
        assert "chart_type" in user

    def test_conversation_history_included(self):
        history = [ConversationTurn(query="prev query", summary="got 42")]
        _, user = build_prompt("next query", "schema", conversation_history=history)
        assert "prev query" in user

    def test_local_mode_shorter_system(self):
        system_cloud, _ = build_prompt("test", "schema", is_local=False)
        system_local, _ = build_prompt("test", "schema", is_local=True)
        assert len(system_local) < len(system_cloud)


class TestParseLLMResponse:
    def test_valid_json(self):
        raw = '{"code": "result = 42", "chart_type": null, "explanation": "The answer"}'
        parsed = parse_llm_response(raw)
        assert parsed.code == "result = 42"
        assert parsed.chart_type is None
        assert parsed.explanation == "The answer"

    def test_json_in_code_fences(self):
        raw = '```json\n{"code": "result = 42", "explanation": "test"}\n```'
        parsed = parse_llm_response(raw)
        assert parsed.code == "result = 42"

    def test_json_with_chart_type(self):
        raw = '{"code": "result = df", "chart_type": "bar", "x_col": "a", "y_col": "b", "explanation": "chart"}'
        parsed = parse_llm_response(raw)
        assert parsed.chart_type == "bar"
        assert parsed.x_col == "a"
        assert parsed.y_col == "b"

    def test_plain_code_fallback(self):
        raw = "result = df['sales'].sum()"
        parsed = parse_llm_response(raw)
        assert "result = df" in parsed.code

    def test_code_in_python_fences(self):
        raw = "```python\nresult = df.shape[0]\n```"
        parsed = parse_llm_response(raw)
        assert "result = df.shape[0]" in parsed.code

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"code": "result = 1", "explanation": "one"}\nDone!'
        parsed = parse_llm_response(raw)
        assert parsed.code == "result = 1"

    def test_empty_response(self):
        parsed = parse_llm_response("")
        assert parsed.code == ""
