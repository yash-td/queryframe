"""Tests for configuration."""

import os
from unittest.mock import patch

import pytest

from queryframe.core.config import QueryFrameConfig
from queryframe.utils.errors import ConfigError


class TestQueryFrameConfig:
    def test_defaults(self):
        config = QueryFrameConfig()
        assert config.provider == "auto"
        assert config.cache_enabled is True
        assert config.sandbox_enabled is True
        assert config.timeout == 30
        assert config.viz_mode == "auto"

    def test_frozen(self):
        config = QueryFrameConfig()
        with pytest.raises(AttributeError):
            config.provider = "openai"

    def test_with_overrides(self):
        config = QueryFrameConfig()
        new_config = config.with_overrides(provider="openai", model="gpt-4o")
        assert new_config.provider == "openai"
        assert new_config.model == "gpt-4o"
        assert config.provider == "auto"  # original unchanged

    def test_validate_valid(self):
        config = QueryFrameConfig(provider="openai")
        config.validate()  # should not raise

    def test_validate_invalid_provider(self):
        config = QueryFrameConfig(provider="invalid")
        with pytest.raises(ConfigError, match="Unknown provider"):
            config.validate()

    def test_validate_invalid_viz(self):
        config = QueryFrameConfig(viz_mode="invalid")
        with pytest.raises(ConfigError, match="Unknown viz_mode"):
            config.validate()

    def test_validate_invalid_timeout(self):
        config = QueryFrameConfig(timeout=0)
        with pytest.raises(ConfigError, match="timeout"):
            config.validate()

    def test_from_env_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=False):
            config = QueryFrameConfig.from_env()
            assert config.api_key == "sk-test123"
            assert config.provider == "openai"

    def test_from_env_anthropic(self):
        env = {"ANTHROPIC_API_KEY": "ant-test123"}
        with patch.dict(os.environ, env, clear=True):
            config = QueryFrameConfig.from_env()
            assert config.provider == "anthropic"

    def test_from_env_explicit_provider(self):
        env = {"QF_PROVIDER": "ollama", "OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=False):
            config = QueryFrameConfig.from_env()
            assert config.provider == "ollama"

    def test_from_env_verbose(self):
        with patch.dict(os.environ, {"QF_VERBOSE": "true"}, clear=False):
            config = QueryFrameConfig.from_env()
            assert config.verbose is True
