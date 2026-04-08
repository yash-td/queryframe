"""Custom exception hierarchy for QueryFrame."""


class QueryFrameError(Exception):
    """Base exception for all QueryFrame errors."""

    help_text: str = ""


class LLMError(QueryFrameError):
    """Error communicating with an LLM provider."""

    help_text = "Check your API key and network connection."


class LLMConnectionError(LLMError):
    """Cannot connect to the LLM provider."""

    help_text = (
        "Cannot reach the LLM server. For cloud providers, check your API key "
        "and network. For local models, ensure Ollama/LM Studio is running."
    )


class LLMResponseError(LLMError):
    """LLM returned an unparseable or invalid response."""

    help_text = "The model returned an invalid response. Try rephrasing your query."


class SandboxError(QueryFrameError):
    """Error during sandboxed code execution."""

    help_text = "The generated code could not be executed safely."


class SandboxTimeoutError(SandboxError):
    """Code execution exceeded the timeout limit."""

    help_text = (
        "The generated code took too long. Try a simpler query or increase "
        "the timeout: qf.configure(timeout=60)"
    )


class SandboxValidationError(SandboxError):
    """Generated code failed safety validation."""

    help_text = (
        "The generated code was blocked for safety reasons. "
        "Try rephrasing your query with simpler operations."
    )


class ValidationError(QueryFrameError):
    """Input validation error."""

    help_text = "Check your input data and query."


class CacheError(QueryFrameError):
    """Error in the caching subsystem."""

    help_text = "Cache error. Try qf.configure(cache_enabled=False) to disable caching."


class ConfigError(QueryFrameError):
    """Invalid configuration."""

    help_text = "Check your QueryFrame configuration. See: qf.configure()"


class ProviderNotFoundError(ConfigError):
    """Requested LLM provider is not available."""

    help_text = (
        "No LLM provider found. Install one:\n"
        "  pip install queryframe[openai]      # OpenAI\n"
        "  pip install queryframe[anthropic]    # Claude\n"
        "  pip install queryframe[gemini]       # Gemini\n"
        "Or set an API key: export OPENAI_API_KEY='sk-...'\n"
        "Or start a local model: ollama serve"
    )
