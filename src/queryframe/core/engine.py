"""Main query engine — orchestrates the full pipeline."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from queryframe.core.config import QueryFrameConfig
from queryframe.core.result import QueryResult
from queryframe.core.schema import compress_schema, extract_schema
from queryframe.llm.base import LLMProvider
from queryframe.llm.prompt.builder import (
    ConversationTurn,
    build_prompt,
    parse_llm_response,
)
from queryframe.llm.prompt.compressor import compress_for_local
from queryframe.llm.registry import auto_detect, get_provider
from queryframe.sandbox.executor import execute_safe
from queryframe.utils.errors import LLMError, QueryFrameError
from queryframe.utils.logger import get_logger

logger = get_logger(__name__)


class QueryEngine:
    """Central orchestrator for QueryFrame queries."""

    def __init__(
        self,
        config: QueryFrameConfig | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self._config = config or QueryFrameConfig.from_env()
        self._config.validate()
        self._provider = provider
        self._conversation: list[ConversationTurn] = []
        self._cache: dict[str, QueryResult] = {}

    @property
    def provider(self) -> LLMProvider:
        """Lazy-initialize the provider."""
        if self._provider is None:
            if self._config.provider == "auto":
                self._provider = auto_detect(
                    model=self._config.model,
                )
            else:
                kwargs: dict[str, Any] = {}
                if self._config.api_key:
                    kwargs["api_key"] = self._config.api_key
                if self._config.model:
                    kwargs["model"] = self._config.model
                if self._config.api_base:
                    kwargs["api_base"] = self._config.api_base
                self._provider = get_provider(self._config.provider, **kwargs)
        return self._provider

    def ask(
        self,
        df: pd.DataFrame,
        query: str,
        viz: str | None = None,
        _previous_code: str | None = None,
        _previous_chart_type: str | None = None,
    ) -> QueryResult:
        """Ask a natural language question about a DataFrame."""
        start = time.perf_counter()

        # Check cache
        schema_info = extract_schema(df, max_samples=self._config.max_sample_rows)
        cache_key = f"{query}:{schema_info.fingerprint}"
        if self._config.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            return QueryResult(
                data=cached.data,
                query=query,
                code=cached.code,
                explanation=cached.explanation,
                provider=cached.provider,
                latency_ms=(time.perf_counter() - start) * 1000,
                chart=cached.chart,
                chart_type=cached.chart_type,
                cached=True,
                _engine=self,
                _df=df,
            )

        # Compress schema
        is_local = self.provider.name in ("ollama", "lmstudio")
        if is_local:
            schema_str = compress_for_local(schema_info, query)
        else:
            schema_str = compress_schema(schema_info, query)

        # Build prompt
        history = self._conversation[-self._config.max_context_turns :]
        system_prompt, user_prompt = build_prompt(
            query=query,
            schema_str=schema_str,
            conversation_history=history,
            is_local=is_local,
        )

        # If re-rendering with different viz, reuse previous code
        if _previous_code is not None:
            code = _previous_code
            chart_type = _previous_chart_type
            explanation = "Re-rendered with different visualization library."
        else:
            # Call LLM with retry
            code, chart_type, explanation = self._call_llm_with_retry(
                df, system_prompt, user_prompt, query
            )

        # Execute code in sandbox
        exec_result = execute_safe(
            code=code,
            df=df,
            timeout=self._config.timeout,
        )

        if exec_result.error:
            logger.warning("Execution error: %s", exec_result.error)
            return QueryResult(
                data=None,
                query=query,
                code=code,
                explanation=f"Error: {exec_result.error}",
                provider=self.provider.name,
                latency_ms=(time.perf_counter() - start) * 1000,
                _engine=self,
                _df=df,
            )

        # Build visualization if requested
        chart = None
        viz_mode = viz or self._config.viz_mode
        if chart_type and viz_mode != "none":
            chart = self._render_chart(
                data=exec_result.data,
                chart_type=chart_type,
                variables=exec_result.variables,
                viz_mode=viz_mode,
            )

        # Store in conversation memory
        summary = str(exec_result.data)[:100] if exec_result.data is not None else "No result"
        self._conversation.append(ConversationTurn(query=query, summary=summary))

        result = QueryResult(
            data=exec_result.data,
            query=query,
            code=code,
            explanation=explanation,
            provider=self.provider.name,
            latency_ms=(time.perf_counter() - start) * 1000,
            chart=chart,
            chart_type=chart_type,
            _engine=self,
            _df=df,
        )

        # Cache the result
        if self._config.cache_enabled:
            self._cache[cache_key] = result

        return result

    def _call_llm_with_retry(
        self,
        df: pd.DataFrame,
        system_prompt: str,
        user_prompt: str,
        query: str,
    ) -> tuple[str, str | None, str]:
        """Call the LLM and retry on execution failures."""
        last_error: str | None = None

        for attempt in range(1 + self._config.max_retries):
            try:
                prompt = user_prompt
                if last_error:
                    prompt += (
                        f"\n\nThe previous code failed with: {last_error}\n"
                        f"Please fix the code and try again."
                    )

                response = self.provider.generate(prompt=prompt, system=system_prompt)
                parsed = parse_llm_response(response.content)

                if not parsed.code:
                    last_error = "LLM returned empty code"
                    continue

                # Try executing to validate
                exec_result = execute_safe(code=parsed.code, df=df, timeout=self._config.timeout)

                if exec_result.error:
                    last_error = exec_result.error
                    logger.info(
                        "Attempt %d failed: %s. Retrying...", attempt + 1, last_error
                    )
                    continue

                return parsed.code, parsed.chart_type, parsed.explanation

            except LLMError:
                raise
            except Exception as e:
                last_error = str(e)
                continue

        # Return the last code even if it failed, let the caller handle the error
        return parsed.code, parsed.chart_type, parsed.explanation

    def _render_chart(
        self,
        data: Any,
        chart_type: str,
        variables: dict[str, Any],
        viz_mode: str,
    ) -> Any:
        """Render a chart using the appropriate visualization library."""
        try:
            from queryframe.viz.selector import select_and_render

            return select_and_render(
                data=data,
                chart_type=chart_type,
                x_col=variables.get("x_col"),
                y_col=variables.get("y_col"),
                title=variables.get("title"),
                viz_mode=viz_mode,
            )
        except ImportError:
            logger.warning("No visualization library available. Install plotly, matplotlib, or altair.")
            return None
        except Exception as e:
            logger.warning("Chart rendering failed: %s", e)
            return None

    def clear_memory(self) -> None:
        """Clear conversation history."""
        self._conversation.clear()

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()
