"""
core/llm/providers/claude_provider.py
========================================

Concrete BaseLLMProvider implementation for Anthropic's Claude API.

Implements `generate()` by calling the `anthropic` SDK and normalizing
the response into LLMResponse (see base_provider.py for why
normalization matters).

Model name, temperature, max_tokens, and timeout come from
config/llm_config.yaml (providers.claude.*) — NOT hardcoded here, so
swapping models doesn't require a code change.

API key comes from config.settings.AppSettings.anthropic_api_key,
which itself reads ANTHROPIC_API_KEY from the environment. This file
never reads os.environ directly (see config/settings.py docstring for
why that boundary is enforced everywhere).
"""

from core.llm.base_provider import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude implementation. See base_provider.py for the
    interface contract this class must satisfy."""

    @property
    def provider_name(self) -> str:
        return "claude"

    # generate() implementation deferred to implementation phase —
    # this file currently defines structure only, per project skeleton scope.
