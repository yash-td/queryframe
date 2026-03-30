"""Anthropic/Claude LLM provider."""

from __future__ import annotations

import time
from dataclasses import dataclass

from queryframe.llm.base import LLMProvider, LLMResponse, TokenUsage
from queryframe.utils.errors import LLMConnectionError, LLMError


@dataclass
class AnthropicProvider:
    """Anthropic Claude API provider."""

    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048

    @property
    def name(self) -> str:
        return "anthropic"

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise LLMError(
                "Anthropic package not installed. Run: pip install queryframe[anthropic]"
            )

        start = time.perf_counter()
        try:
            client = Anthropic(api_key=self.api_key)

            kwargs: dict = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)
        except Exception as e:
            if "connection" in str(e).lower():
                raise LLMConnectionError(f"Cannot connect to Anthropic: {e}")
            raise LLMError(f"Anthropic API error: {e}")

        latency = (time.perf_counter() - start) * 1000
        content = response.content[0].text if response.content else ""
        usage = response.usage

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
            model=response.model,
            latency_ms=latency,
        )
