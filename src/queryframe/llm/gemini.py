"""Google Gemini LLM provider."""

from __future__ import annotations

import time
from dataclasses import dataclass

from queryframe.llm.base import LLMProvider, LLMResponse, TokenUsage
from queryframe.utils.errors import LLMConnectionError, LLMError


@dataclass
class GeminiProvider:
    """Google Gemini API provider."""

    api_key: str
    model: str = "gemini-2.0-flash"
    max_tokens: int = 2048

    @property
    def name(self) -> str:
        return "gemini"

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        try:
            from google import genai
        except ImportError:
            raise LLMError(
                "Google GenAI package not installed. Run: pip install queryframe[gemini]"
            )

        start = time.perf_counter()
        try:
            client = genai.Client(api_key=self.api_key)

            full_prompt = prompt
            if system:
                full_prompt = f"{system}\n\n{prompt}"

            response = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        except Exception as e:
            if "connection" in str(e).lower():
                raise LLMConnectionError(f"Cannot connect to Gemini: {e}")
            raise LLMError(f"Gemini API error: {e}")

        latency = (time.perf_counter() - start) * 1000
        content = response.text or ""

        # Extract token counts if available
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(meta, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=self.model,
            latency_ms=latency,
        )
