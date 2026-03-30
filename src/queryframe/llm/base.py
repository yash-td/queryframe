"""Abstract LLM provider protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TokenUsage:
    """Token usage statistics from an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    usage: TokenUsage
    model: str
    latency_ms: float


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement."""

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """Generate a completion synchronously."""
        ...
