"""
core/llm/providers/groq_provider.py
======================================

Concrete BaseLLMProvider implementation for Groq's API. The only LLM
provider with a working implementation in V1 — see config/settings.py
and core/llm/llm_client_factory.py for how Groq became the default and
why the other three providers remain registered stubs rather than
removed entirely.

WHY GROQ FOR V1
-----------------
Per the original architecture discussion: Groq is useful as a fast,
low-latency provider — a reasonable single-provider choice for getting
the whole core/ pipeline (SQL generation, validation, execution,
insights) working end-to-end before adding the cost/latency tradeoffs
of multiple providers. Nothing about this implementation is Groq-
specific beyond this one file — sql_generator.py, insight_generator.py,
and intent_classifier.py all depend on BaseLLMProvider, not on Groq.

GROUNDING NOTE: the request/response shapes below were verified
directly against the installed `groq` Python SDK (v1.4.0), not
assumed from documentation. Specifically:
  - `max_tokens` is a deprecated parameter name in the current SDK;
    `max_completion_tokens` is the current one. Both still work, but
    this implementation uses the current parameter.
  - Response content path: `response.choices[0].message.content`.
  - Token usage: `response.usage.prompt_tokens` /
    `response.usage.completion_tokens`.
  - The SDK's exception hierarchy (groq.APIError and subclasses:
    APIConnectionError, APITimeoutError, RateLimitError,
    AuthenticationError, etc.) is mapped to this project's own
    exceptions/domain_exceptions.py types in generate() below, so
    nothing downstream of this file ever needs to import or catch a
    groq.* exception directly.
"""

from __future__ import annotations

import time

import groq

from core.llm.base_provider import BaseLLMProvider, LLMResponse
from core.llm.llm_config_loader import ResolvedLLMConfig
from exceptions.domain_exceptions import (
    LLMAPIError,
    LLMResponseParsingError,
    LLMTimeoutError,
)
from logging_setup.logger import LogCategory, LogFields, get_logger

logger = get_logger(__name__)


class GroqProvider(BaseLLMProvider):
    """
    Groq implementation of BaseLLMProvider.

    Construction takes an already-resolved config (ResolvedLLMConfig)
    and an API key, rather than reading config.settings or
    config/llm_config.yaml itself — see llm_config_loader.py's
    docstring on why config resolution is centralized outside
    individual providers. This also makes GroqProvider trivially
    testable: construct it with a fake/test API key pointed at a
    mocked transport, no environment setup required.
    """

    def __init__(self, api_key: str, config: ResolvedLLMConfig) -> None:
        """
        Args:
            api_key: Groq API key. Passed in explicitly (never read
                from os.environ inside this class) — see
                llm_client_factory.py for where this is sourced from
                config.settings.AppSettings.groq_api_key.
            config: resolved model/temperature/max_tokens/timeout,
                from llm_config_loader.resolve_groq_config().
        """
        self._config = config
        self._client = groq.Groq(api_key=api_key, timeout=config.timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "groq"

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Sends a single-turn chat completion request to Groq.

        Args:
            system_prompt: system-role instruction text.
            user_message: user-role message content.
            temperature: overrides this provider's configured default
                (self._config.temperature) for this one call. Used by
                callers needing a different temperature for a specific
                task — e.g. insight generation wants a higher
                temperature than SQL generation, per
                config/llm_config.yaml's insight_generation section —
                without constructing a second GroqProvider instance
                just to change one number.
            max_tokens: same override pattern as temperature, for
                this provider's configured max_tokens default.

        Returns:
            Normalized LLMResponse (see base_provider.py).

        Raises:
            LLMTimeoutError: request exceeded the configured timeout.
            LLMAPIError: any other API-level failure (auth, rate
                limit, connection error, 5xx).
            LLMResponseParsingError: the response didn't contain the
                expected structure (e.g. empty choices list) — should
                not normally happen, but is handled explicitly rather
                than letting an IndexError/AttributeError propagate
                as an unhandled exception.
        """
        effective_temperature = temperature if temperature is not None else self._config.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        log = logger.bind(
            category=LogCategory.LLM_CALL,
            **{LogFields.LLM_PROVIDER: self.provider_name, LogFields.LLM_MODEL: self._config.model},
        )

        start = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=effective_temperature,
                max_completion_tokens=effective_max_tokens,
            )
        except groq.APITimeoutError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            log.error("llm_call_timeout", **{LogFields.LATENCY_MS: round(elapsed_ms, 2)})
            raise LLMTimeoutError(
                message=f"Groq API call timed out after {self._config.timeout_seconds}s: {exc}",
                user_message="The AI is taking too long to respond. Please try again.",
            ) from exc
        except groq.AuthenticationError as exc:
            log.error("llm_call_auth_failed", error=str(exc))
            raise LLMAPIError(
                message=f"Groq authentication failed — check GROQ_API_KEY: {exc}",
                user_message="There's a configuration issue with the AI service. Contact your administrator.",
            ) from exc
        except groq.RateLimitError as exc:
            log.error("llm_call_rate_limited", error=str(exc))
            raise LLMAPIError(
                message=f"Groq API rate limit exceeded: {exc}",
                user_message="The AI service is currently busy. Please try again in a moment.",
            ) from exc
        except groq.APIError as exc:
            # Catches the remaining groq.APIError subclasses
            # (APIConnectionError, BadRequestError, InternalServerError,
            # etc.) under one handler, since they all map to the same
            # generic "the call failed" outcome for this project's
            # purposes — see LLMAPIError's docstring. Authentication
            # and rate-limit errors are caught above specifically
            # because they warrant distinct user_message text.
            elapsed_ms = (time.monotonic() - start) * 1000
            log.error(
                "llm_call_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                **{LogFields.LATENCY_MS: round(elapsed_ms, 2)},
            )
            raise LLMAPIError(
                message=f"Groq API call failed ({type(exc).__name__}): {exc}",
                user_message="The AI service encountered an error. Please try again.",
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000

        try:
            content = response.choices[0].message.content
            if content is None:
                raise LLMResponseParsingError(
                    message="Groq response message content was None.",
                    user_message="The AI returned an empty response. Please try again.",
                )

            normalized = LLMResponse(
                content=content,
                raw_provider_response=response,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=elapsed_ms,
                model=response.model,
            )
        except (IndexError, AttributeError) as exc:
            log.error("llm_response_parsing_failed", error=str(exc))
            raise LLMResponseParsingError(
                message=f"Could not parse Groq response into expected structure: {exc}",
                user_message="The AI returned an unexpected response. Please try again.",
            ) from exc

        log.info(
            "llm_call_succeeded",
            **{
                LogFields.LATENCY_MS: round(elapsed_ms, 2),
                LogFields.TOKEN_COUNT_INPUT: normalized.input_tokens,
                LogFields.TOKEN_COUNT_OUTPUT: normalized.output_tokens,
            },
        )

        return normalized
