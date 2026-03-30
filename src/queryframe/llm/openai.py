"""OpenAI LLM provider."""

from __future__ import annotations

import time
from dataclasses import dataclass

from queryframe.llm.base import LLMProvider, LLMResponse, TokenUsage
from queryframe.utils.errors import LLMConnectionError, LLMError


@dataclass
class OpenAIProvider:
    """OpenAI API provider."""

    api_key: str
    model: str = "gpt-4o-mini"
    api_base: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048

    @property
    def name(self) -> str:
        return "openai"

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError(
                "OpenAI package not installed. Run: pip install queryframe[openai]"
            )

        start = time.perf_counter()
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            if "connection" in str(e).lower():
                raise LLMConnectionError(f"Cannot connect to OpenAI: {e}")
            raise LLMError(f"OpenAI API error: {e}")

        latency = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            model=response.model,
            latency_ms=latency,
        )
