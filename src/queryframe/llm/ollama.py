"""Ollama local model provider."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from queryframe.llm.base import LLMProvider, LLMResponse, TokenUsage
from queryframe.utils.errors import LLMConnectionError, LLMError


@dataclass
class OllamaProvider:
    """Ollama local model provider via HTTP API."""

    model: str = "llama3.1"
    api_base: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 2048
    # Accept any extra kwargs silently
    api_key: str | None = None

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        try:
            import httpx
        except ImportError:
            # Fallback to urllib
            return self._generate_urllib(prompt, system)

        start = time.perf_counter()
        url = f"{self.api_base}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "format": "json",
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.api_base}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except Exception as e:
            raise LLMError(f"Ollama API error: {e}")

        latency = (time.perf_counter() - start) * 1000
        content = data.get("response", "")

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            model=data.get("model", self.model),
            latency_ms=latency,
        )

    def _generate_urllib(self, prompt: str, system: str = "") -> LLMResponse:
        """Fallback using urllib when httpx is not installed."""
        import urllib.request

        start = time.perf_counter()
        url = f"{self.api_base}/api/generate"

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "format": "json",
        }).encode()

        try:
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except ConnectionRefusedError:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.api_base}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except Exception as e:
            raise LLMError(f"Ollama API error: {e}")

        latency = (time.perf_counter() - start) * 1000
        content = data.get("response", "")

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            model=data.get("model", self.model),
            latency_ms=latency,
        )

    def list_models(self) -> list[str]:
        """List available models from Ollama."""
        import urllib.request

        try:
            url = f"{self.api_base}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
