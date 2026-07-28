"""
core/llm/llm_client_factory.py
=================================

Factory responsible for instantiating the correct BaseLLMProvider
based on configuration, and ONLY based on configuration.

WHY A FACTORY INSTEAD OF DIRECT INSTANTIATION EVERYWHERE
-------------------------------------------------------------
Without this, every module that needs an LLM (sql_generator.py,
insight_generator.py, intent_classifier.py) would need its own
if/elif chain over `settings.default_llm_provider`. With the factory,
provider selection happens in exactly one place:

    from core.llm.llm_client_factory import get_llm_provider

    provider = get_llm_provider()   # reads settings, returns configured provider
    response = provider.generate(system_prompt=..., user_message=...)

Callers depend on BaseLLMProvider (the abstract interface), never on
GroqProvider/ClaudeProvider/etc. directly.

V1 SCOPE: GROQ ONLY
----------------------
Only LLMProvider.GROQ has a working entry in _PROVIDER_BUILDERS below.
The other three enum values (claude/openai/gemini) remain on the
LLMProvider enum and their provider classes remain as stubs in
core/llm/providers/ — but requesting them here raises
ConfigurationError with a clear, specific message, rather than either
silently falling back to Groq (which would hide a real configuration
mistake) or raising a generic NotImplementedError deep inside an
unfinished provider class. This is a deliberate fail-fast choice: a
deployment that sets DEFAULT_LLM_PROVIDER=openai today should find out
immediately at startup (see app/main.py's bootstrap(), which is
expected to call get_llm_provider() once eagerly) that OpenAI isn't
available yet, not three turns into a conversation when the first SQL
generation call is attempted.

ADDING A NEW PROVIDER LATER
------------------------------
1. Implement generate() in the corresponding core/llm/providers/*.py
   class (e.g. ClaudeProvider), following groq_provider.py as the
   reference implementation — same constructor shape (api_key,
   ResolvedLLMConfig), same exception-mapping pattern at the bottom of
   generate().
2. Add a `resolve_<provider>_config()` function to
   core/llm/llm_config_loader.py, mirroring resolve_groq_config().
3. Add one entry to _PROVIDER_BUILDERS below. Nothing else in the
   codebase changes — every caller already depends on
   BaseLLMProvider, not on a specific class.
"""

from __future__ import annotations

from collections.abc import Callable

from config.settings import LLMProvider, get_settings
from core.llm.base_provider import BaseLLMProvider
from core.llm.llm_config_loader import resolve_groq_config
from core.llm.providers.groq_provider import GroqProvider
from exceptions.domain_exceptions import ConfigurationError
from logging_setup.logger import LogCategory, get_logger

logger = get_logger(__name__)


def _build_groq_provider() -> BaseLLMProvider:
    settings = get_settings()
    if settings.groq_api_key is None:
        raise ConfigurationError(
            message="DEFAULT_LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set.",
            user_message="The AI service isn't configured. Contact your administrator.",
        )

    config = resolve_groq_config()
    return GroqProvider(api_key=settings.groq_api_key.get_secret_value(), config=config)


def _build_unimplemented_provider(provider_name: str) -> BaseLLMProvider:
    """
    Shared raiser for any LLMProvider value not yet backed by a
    working implementation (currently: claude, openai, gemini).
    Centralized here rather than duplicated per provider so the
    error message stays consistent and is defined in exactly one
    place as more providers move from "stub" to "raises this" to
    "actually implemented."
    """
    raise ConfigurationError(
        message=f"LLM provider '{provider_name}' is not yet implemented in this version.",
        user_message=(
            f"The '{provider_name}' AI provider isn't available yet. "
            "Groq is currently the only supported provider."
        ),
    )


# Dispatch table: LLMProvider -> zero-argument builder function.
# Builders (not pre-built instances) because constructing a provider
# requires reading settings/config at call time, not at module import
# time — settings may not be finalized yet when this module is first
# imported (e.g. during test collection).
_PROVIDER_BUILDERS: dict[LLMProvider, Callable[[], BaseLLMProvider]] = {
    LLMProvider.GROQ: _build_groq_provider,
    LLMProvider.CLAUDE: lambda: _build_unimplemented_provider("claude"),
    LLMProvider.OPENAI: lambda: _build_unimplemented_provider("openai"),
    LLMProvider.GEMINI: lambda: _build_unimplemented_provider("gemini"),
}

# Cache of constructed providers, keyed by resolved LLMProvider.
#
# WHY THIS EXISTS (added during the production hardening pass):
# Without this, every call to get_llm_provider() — once per
# generate_sql() call, once per generate_insight() call, i.e. at least
# twice per user question — constructed a brand new groq.Groq() client
# from scratch, each with its own fresh HTTP connection pool. Over a
# multi-turn conversation this meant repeatedly paying TCP/TLS
# handshake cost instead of reusing keep-alive connections, for no
# benefit — the provider is stateless-safe to reuse (it holds only a
# configured HTTP client, no per-call mutable state).
#
# Cached by resolved_provider (not raw `provider` argument), so
# get_llm_provider() and get_llm_provider(LLMProvider.GROQ) share the
# same cached instance when GROQ is the configured default — they
# resolve to the same provider, so they should hit the same client.
_provider_cache: dict[LLMProvider, BaseLLMProvider] = {}


def get_llm_provider(provider: LLMProvider | None = None) -> BaseLLMProvider:
    """
    Returns a cached, constructed provider implementing BaseLLMProvider.

    Args:
        provider: explicit provider to use. If None, falls back to
            settings.default_llm_provider (Groq, in V1's default
            configuration — see config/settings.py).

    Returns:
        A constructed, ready-to-use BaseLLMProvider, reused across
        calls for the same resolved provider (see _provider_cache
        above). Note this does NOT make a network call to verify the
        API key works — only that it's present. The first real
        generate() call is where an invalid key would surface as
        LLMAPIError (groq.AuthenticationError, specifically — see
        groq_provider.py).

    Raises:
        ConfigurationError: the resolved provider has no API key
            configured, OR the resolved provider isn't implemented yet
            (claude/openai/gemini in V1 — see module docstring).
    """
    settings = get_settings()
    resolved_provider = provider or settings.default_llm_provider

    if resolved_provider in _provider_cache:
        return _provider_cache[resolved_provider]

    builder = _PROVIDER_BUILDERS.get(resolved_provider)
    if builder is None:
        # Defensive: only reachable if LLMProvider gains a new enum
        # value without a corresponding _PROVIDER_BUILDERS entry.
        logger.error(
            "llm_provider_not_registered",
            category=LogCategory.LLM_CALL,
            provider=resolved_provider.value,
        )
        raise ConfigurationError(
            message=f"No builder registered for LLM provider '{resolved_provider.value}'.",
            user_message="The AI service isn't configured correctly. Contact your administrator.",
        )

    constructed = builder()
    _provider_cache[resolved_provider] = constructed
    return constructed


def clear_provider_cache() -> None:
    """
    Clears the cached provider instances. Used by tests that need a
    fresh provider construction (e.g. to verify ConfigurationError is
    raised again after simulating a missing API key), and available
    for any future scenario where settings change at runtime and a
    stale cached client needs to be discarded.
    """
    _provider_cache.clear()
