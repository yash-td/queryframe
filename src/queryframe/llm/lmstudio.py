"""LM Studio local model provider (OpenAI-compatible API)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from queryframe.llm.base import LLMProvider, LLMResponse, TokenUsage
from queryframe.utils.errors import LLMConnectionError, LLMError


@dataclass
class LMStudioProvider:
    """LM Studio provider using OpenAI-compatible API at localhost:1234."""

    model: str = "local-model"
    api_base: str = "http://localhost:1234/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    # Accept any extra kwargs silently
    api_key: str | None = None

    @property
    def name(self) -> str:
        return "lmstudio"

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError(
                "OpenAI package not installed (required for LM Studio compatibility). "
                "Run: pip install queryframe[lmstudio]"
            )

        start = time.perf_counter()
        try:
            # LM Studio uses OpenAI-compatible API, no real API key needed
            client = OpenAI(
                api_key=self.api_key or "lm-studio",
                base_url=self.api_base,
            )
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "connection" in err_str or "refused" in err_str:
                raise LLMConnectionError(
                    f"Cannot connect to LM Studio at {self.api_base}. "
                    "Is LM Studio running with a model loaded?"
                )
            raise LLMError(f"LM Studio API error: {e}")

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
            model=response.model or self.model,
            latency_ms=latency,
        )
