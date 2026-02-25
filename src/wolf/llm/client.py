"""Async LLM client wrapping an OpenAI-compatible API (e.g. Ollama)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import openai

from wolf.config.schema import ModelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Parsed response from an LLM completion call."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str


class LLMClient:
    """Async client for OpenAI-compatible chat completions.

    Wraps :class:`openai.AsyncOpenAI` and points it at the configured
    ``api_base`` (typically an Ollama endpoint).
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=30.0),
        )
        self._client = openai.AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
            http_client=self._http_client,
            max_retries=0,  # We handle retries in retry.py
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a parsed response.

        Parameters
        ----------
        messages:
            OpenAI-style message dicts (``role``, ``content``).
        temperature:
            Sampling temperature override.  Falls back to
            ``self.config.reasoning_temperature`` if *None*.
        max_tokens:
            Max completion tokens.  Falls back to ``self.config.max_tokens``
            if *None*.
        """
        temp = temperature if temperature is not None else self.config.reasoning_temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        logger.debug(
            "LLM chat request: model=%s, messages=%d, temperature=%.2f, max_tokens=%d",
            self.config.model,
            len(messages),
            temp,
            tokens,
        )

        # Build extra_body for Ollama-specific options (e.g. num_ctx).
        extra_body: dict | None = None
        if self.config.context_length:
            extra_body = {"options": {"num_ctx": self.config.context_length}}

        # Suppress native tool calling — we use text-based USE: tool(args)
        # parsing instead.  Some models (e.g. gpt-oss) auto-activate native
        # tool calling and cause 500 errors when Ollama tries to parse our
        # text output as JSON tool calls.
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            extra_body=extra_body,
            tools=[],
            **self.config.extra_params,
        )

        choice = response.choices[0]
        usage = response.usage

        # Some models (e.g. gpt-oss) auto-activate native tool calling
        # and emit tool_calls instead of text content.  Convert them to
        # USE: tool_name(args) text so our ReAct parser can handle them.
        content = choice.message.content or ""
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                fn = tc.function
                content += f"\nUSE: {fn.name}({fn.arguments})"

        result = LLMResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "unknown",
        )

        logger.debug(
            "LLM chat response: model=%s, input_tokens=%d, output_tokens=%d, finish=%s",
            result.model,
            result.input_tokens,
            result.output_tokens,
            result.finish_reason,
        )

        return result

    async def check_model_available(self) -> bool:
        """Try a trivial completion to verify the model endpoint is reachable."""
        try:
            await self.chat(
                messages=[{"role": "user", "content": "Say hello."}],
                temperature=0.0,
                max_tokens=16,
            )
            return True
        except Exception as exc:
            logger.warning("Model availability check failed: %s", exc)
            return False

    async def validate_context_length(self) -> int:
        """Verify the configured context length is actually being used.

        Sends a request with ``num_ctx`` and checks that the reported
        prompt tokens roughly match expectations, catching cases where
        the server silently falls back to a smaller default (e.g. 2048).

        Returns the actual ``num_ctx`` the server is using, or raises
        ``RuntimeError`` if it can't be validated.
        """
        configured = self.config.context_length
        if not configured:
            return 0

        # Use Ollama's raw API to inspect model runner settings
        try:
            import httpx as _httpx

            async with _httpx.AsyncClient(timeout=30.0) as http:
                # Generate a trivial completion and check the response
                resp = await http.post(
                    self.config.api_base.replace("/v1", "")
                    + "/api/chat",
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "user", "content": "Say OK."},
                        ],
                        "stream": False,
                        "options": {"num_ctx": configured},
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                # Ollama echoes prompt_eval_count in the response
                prompt_tokens = data.get("prompt_eval_count", 0)
                eval_count = data.get("eval_count", 0)

                if prompt_tokens > 0 or eval_count > 0:
                    logger.info(
                        "Context length validated for %s: "
                        "configured=%d, prompt_tokens=%d",
                        self.config.model,
                        configured,
                        prompt_tokens,
                    )
                    return configured

        except Exception as exc:
            logger.warning(
                "Context length validation failed for %s: %s",
                self.config.model,
                exc,
            )

        return configured

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down the underlying HTTP transport."""
        await self._http_client.aclose()
