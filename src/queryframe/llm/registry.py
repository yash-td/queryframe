"""Provider auto-detection and registry."""

from __future__ import annotations

import os
from typing import Callable

from queryframe.llm.base import LLMProvider
from queryframe.utils.errors import ProviderNotFoundError
from queryframe.utils.logger import get_logger

logger = get_logger(__name__)

ProviderFactory = Callable[..., LLMProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory."""
    _REGISTRY[name] = factory


def get_provider(name: str, **kwargs: object) -> LLMProvider:
    """Get a provider by name."""
    if name not in _REGISTRY:
        raise ProviderNotFoundError(
            f"Provider '{name}' not registered. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


def auto_detect(**kwargs: object) -> LLMProvider:
    """Auto-detect the best available provider.

    Checks environment variables first, then tries local providers.
    """
    # Check cloud API keys
    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Auto-detected OpenAI provider via OPENAI_API_KEY")
        return get_provider("openai", api_key=os.environ["OPENAI_API_KEY"], **kwargs)

    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Auto-detected Anthropic provider via ANTHROPIC_API_KEY")
        return get_provider("anthropic", api_key=os.environ["ANTHROPIC_API_KEY"], **kwargs)

    if os.environ.get("GOOGLE_API_KEY"):
        logger.info("Auto-detected Gemini provider via GOOGLE_API_KEY")
        return get_provider("gemini", api_key=os.environ["GOOGLE_API_KEY"], **kwargs)

    # Check local providers
    if _check_local_server("http://localhost:11434"):
        logger.info("Auto-detected Ollama at localhost:11434")
        return get_provider("ollama", **kwargs)

    if _check_local_server("http://localhost:1234"):
        logger.info("Auto-detected LM Studio at localhost:1234")
        return get_provider("lmstudio", **kwargs)

    raise ProviderNotFoundError(
        "No LLM provider found. Set an API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "GOOGLE_API_KEY) or start a local model server (Ollama, LM Studio)."
    )


def _check_local_server(base_url: str) -> bool:
    """Check if a local server is running."""
    try:
        import urllib.request

        req = urllib.request.Request(base_url, method="HEAD")
        urllib.request.urlopen(req, timeout=1)
        return True
    except Exception:
        return False


# Register built-in providers (lazy imports to avoid requiring all deps)
def _openai_factory(**kwargs: object) -> LLMProvider:
    from queryframe.llm.openai import OpenAIProvider

    return OpenAIProvider(**kwargs)  # type: ignore[arg-type]


def _anthropic_factory(**kwargs: object) -> LLMProvider:
    from queryframe.llm.anthropic import AnthropicProvider

    return AnthropicProvider(**kwargs)  # type: ignore[arg-type]


def _gemini_factory(**kwargs: object) -> LLMProvider:
    from queryframe.llm.gemini import GeminiProvider

    return GeminiProvider(**kwargs)  # type: ignore[arg-type]


def _ollama_factory(**kwargs: object) -> LLMProvider:
    from queryframe.llm.ollama import OllamaProvider

    return OllamaProvider(**kwargs)  # type: ignore[arg-type]


def _lmstudio_factory(**kwargs: object) -> LLMProvider:
    from queryframe.llm.lmstudio import LMStudioProvider

    return LMStudioProvider(**kwargs)  # type: ignore[arg-type]


register("openai", _openai_factory)
register("anthropic", _anthropic_factory)
register("gemini", _gemini_factory)
register("ollama", _ollama_factory)
register("lmstudio", _lmstudio_factory)
