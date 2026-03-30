"""Custom exception hierarchy for QueryFrame."""


class QueryFrameError(Exception):
    """Base exception for all QueryFrame errors."""


class LLMError(QueryFrameError):
    """Error communicating with an LLM provider."""


class LLMConnectionError(LLMError):
    """Cannot connect to the LLM provider."""


class LLMResponseError(LLMError):
    """LLM returned an unparseable or invalid response."""


class SandboxError(QueryFrameError):
    """Error during sandboxed code execution."""


class SandboxTimeoutError(SandboxError):
    """Code execution exceeded the timeout limit."""


class SandboxValidationError(SandboxError):
    """Generated code failed safety validation."""


class ValidationError(QueryFrameError):
    """Input validation error."""


class CacheError(QueryFrameError):
    """Error in the caching subsystem."""


class ConfigError(QueryFrameError):
    """Invalid configuration."""


class ProviderNotFoundError(ConfigError):
    """Requested LLM provider is not available."""
