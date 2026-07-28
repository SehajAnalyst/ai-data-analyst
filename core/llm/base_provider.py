"""
core/llm/base_provider.py
============================

Abstract interface that every LLM provider (Claude, OpenAI, Gemini,
Groq) must implement.

WHY THIS EXISTS
----------------
This is the single most important file for satisfying the "configurable
LLM" requirement without duplicating logic four times. Every other
module in core/ (sql_generator.py, insight_generator.py,
intent_classifier.py) depends on THIS interface, never on a specific
provider's SDK directly. That means:

  - sql_generator.py has zero knowledge of whether it's talking to
    Claude or Groq.
  - Adding a 5th provider later means writing one new class that
    implements this interface — nothing else in the codebase changes.
  - Tests can use a FakeLLMProvider implementing this same interface,
    with no network calls, no API keys, fully deterministic.

This is the Strategy pattern, applied at the provider boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    Normalized response shape returned by every provider, regardless
    of how that provider's native SDK structures its output.

    WHY NORMALIZE: Claude, OpenAI, Gemini, and Groq all return
    differently-shaped response objects natively. If sql_generator.py
    had to handle each shape, every prompt-handling change would
    require four conditional branches. Providers translate their own
    SDK's response into this shape internally; nothing downstream
    needs to know the provider-specific structure.
    """

    content: str
    raw_provider_response: object  # kept for debugging/logging only
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class BaseLLMProvider(ABC):
    """
    Every concrete provider (ClaudeProvider, OpenAIProvider, etc.)
    extends this and implements `generate()`.

    DESIGN NOTE: this interface is intentionally narrow — just
    "send messages, get normalized text back." It deliberately does
    NOT expose provider-specific features (e.g. a Claude-only beta
    header) through this shared interface, because doing so would
    leak provider-specific behavior into code that's supposed to be
    provider-agnostic. If a provider-specific feature becomes
    necessary, it belongs in that provider's own class as an
    additional, explicitly-named method — not bolted onto this
    shared contract.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send a single-turn request to the underlying LLM API and
        return a normalized LLMResponse.

        Args:
            system_prompt: system-role instruction text.
            user_message: user-role message content.
            temperature: overrides the provider's configured default
                for this one call. None means "use the provider's
                configured default" (resolved at construction time
                from config/llm_config.yaml + settings overrides —
                see core/llm/llm_config_loader.py). Per-call overrides
                exist because different tasks legitimately want
                different temperatures (e.g. SQL generation wants
                deterministic 0.0; insight generation wants some
                flexibility) without constructing a second provider
                instance just to change one number.
            max_tokens: same override pattern as temperature.

        Conversation history (if any) is the caller's responsibility
        to fold into `user_message` or a structured equivalent —
        kept simple here deliberately; multi-turn-native handling can
        be added later if a real need emerges, rather than guessed at
        now.

        Raises:
            LLMAPIError: underlying API call failed.
            LLMTimeoutError: call exceeded configured timeout.
            LLMResponseParsingError: response could not be normalized.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier, e.g. 'claude', used in logging."""
        raise NotImplementedError
