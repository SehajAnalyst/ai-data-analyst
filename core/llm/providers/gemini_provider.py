"""
core/llm/providers/gemini_provider.py
========================================

Concrete BaseLLMProvider implementation for Google's Gemini API.
See claude_provider.py for the documentation pattern this follows.
"""

from core.llm.base_provider import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Gemini implementation. See base_provider.py for the interface
    contract this class must satisfy."""

    @property
    def provider_name(self) -> str:
        return "gemini"

    # generate() implementation deferred to implementation phase.
