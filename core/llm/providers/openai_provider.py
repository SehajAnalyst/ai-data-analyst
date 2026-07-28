"""
core/llm/providers/openai_provider.py
========================================

Concrete BaseLLMProvider implementation for OpenAI's API.
See claude_provider.py for the documentation pattern this follows —
identical structure, different SDK underneath.
"""

from core.llm.base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI implementation. See base_provider.py for the interface
    contract this class must satisfy."""

    @property
    def provider_name(self) -> str:
        return "openai"

    # generate() implementation deferred to implementation phase.
