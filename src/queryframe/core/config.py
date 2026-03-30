"""Global configuration for QueryFrame."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from queryframe.utils.errors import ConfigError


@dataclass(frozen=True)
class QueryFrameConfig:
    """Immutable configuration for QueryFrame.

    Can be created directly or via `from_env()` to read QF_* environment variables.
    """

    provider: str = "auto"
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    cache_enabled: bool = True
    sandbox_enabled: bool = True
    timeout: int = 30
    viz_mode: Literal["auto", "plotly", "matplotlib", "altair"] = "auto"
    max_retries: int = 2
    verbose: bool = False
    max_sample_rows: int = 3
    max_context_turns: int = 3

    @classmethod
    def from_env(cls) -> QueryFrameConfig:
        """Create config from QF_* environment variables."""
        kwargs: dict = {}
        env_map = {
            "QF_PROVIDER": "provider",
            "QF_MODEL": "model",
            "QF_API_KEY": "api_key",
            "QF_API_BASE": "api_base",
            "QF_TIMEOUT": "timeout",
            "QF_VIZ": "viz_mode",
            "QF_MAX_RETRIES": "max_retries",
            "QF_VERBOSE": "verbose",
        }
        for env_var, field_name in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                if field_name in ("timeout", "max_retries"):
                    kwargs[field_name] = int(val)
                elif field_name == "verbose":
                    kwargs[field_name] = val.lower() in ("1", "true", "yes")
                else:
                    kwargs[field_name] = val

        # Auto-detect API keys from standard env vars
        if "api_key" not in kwargs:
            for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
                key = os.environ.get(env_var)
                if key:
                    kwargs["api_key"] = key
                    if "provider" not in kwargs:
                        provider_map = {
                            "OPENAI_API_KEY": "openai",
                            "ANTHROPIC_API_KEY": "anthropic",
                            "GOOGLE_API_KEY": "gemini",
                        }
                        kwargs["provider"] = provider_map[env_var]
                    break

        return cls(**kwargs)

    def with_overrides(self, **kwargs: object) -> QueryFrameConfig:
        """Return a new config with the given overrides applied."""
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return QueryFrameConfig(**current)

    def validate(self) -> None:
        """Validate the configuration, raising ConfigError if invalid."""
        valid_providers = {"auto", "openai", "anthropic", "gemini", "ollama", "lmstudio"}
        if self.provider not in valid_providers:
            raise ConfigError(
                f"Unknown provider '{self.provider}'. Valid: {', '.join(sorted(valid_providers))}"
            )
        valid_viz = {"auto", "plotly", "matplotlib", "altair"}
        if self.viz_mode not in valid_viz:
            raise ConfigError(
                f"Unknown viz_mode '{self.viz_mode}'. Valid: {', '.join(sorted(valid_viz))}"
            )
        if self.timeout < 1:
            raise ConfigError("timeout must be >= 1")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
