"""
core/llm/llm_config_loader.py
================================

Resolves the final, effective configuration (model name, temperature,
max_tokens, timeout) for a given LLM provider, by merging two sources:

  1. config/llm_config.yaml — non-secret, version-controlled defaults
     per provider (see that file's own header comment for why it's
     YAML and not Python).
  2. config.settings.AppSettings — environment-variable overrides
     (e.g. GROQ_MODEL, GROQ_TEMPERATURE) that, when set, take
     precedence over the YAML default for that specific field.

WHY A SEPARATE LOADER MODULE, NOT INLINE IN THE FACTORY OR PROVIDER
-------------------------------------------------------------------------
The merge logic (env override wins, else YAML default, else a hard
fallback) is the same shape for every provider and every tunable field.
Putting it inline in llm_client_factory.py would mean repeating that
same "check env, fall back to YAML" pattern once per provider as more
providers get implemented. Centralizing it here means
GroqProvider — and ClaudeProvider/OpenAIProvider/GeminiProvider later
— all resolve their config the same documented way, from one place.

WHY YAML IS LOADED HERE RATHER THAN VIA config.settings
-------------------------------------------------------------------------
config.settings.AppSettings is specifically for environment-variable-
sourced configuration (see that module's docstring on why
os.getenv() calls are centralized there). llm_config.yaml is a
different configuration source — a tracked file, not the environment —
and parsing it doesn't belong inside the pydantic-settings model meant
for env vars. Keeping them as two distinct loaders, merged explicitly
in this one place, keeps each loader doing one well-defined job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config.settings import get_settings
from exceptions.domain_exceptions import ConfigurationError
from logging_setup.logger import get_logger

logger = get_logger(__name__)

_LLM_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "llm_config.yaml"

_yaml_cache: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedLLMConfig:
    """
    The final, effective configuration for one provider, after merging
    YAML defaults with any environment-variable overrides. This is
    what gets passed into a provider's constructor — providers never
    read config/llm_config.yaml or config.settings directly themselves
    (see groq_provider.py), so they stay testable with a plain
    ResolvedLLMConfig instance and no environment/file dependencies.
    """

    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class ResolvedInsightConfig:
    """Effective config for the business-insights LLM call — a
    separate, simpler shape from ResolvedLLMConfig since insight
    generation reuses whichever provider is already configured
    (Groq in V1) and only needs its own temperature/max_tokens,
    not a model name or timeout."""

    temperature: float
    max_tokens: int


def _load_yaml_config() -> dict[str, Any]:
    """
    Loads and caches config/llm_config.yaml for the process lifetime.
    Cached at module level (not via functools.lru_cache on a function
    with no arguments, which would work too, but a plain module-level
    cache is more obviously inspectable/clearable in tests) since this
    file changes only via deployment, never at runtime.
    """
    global _yaml_cache
    if _yaml_cache is not None:
        return _yaml_cache

    if not _LLM_CONFIG_PATH.is_file():
        raise ConfigurationError(
            message=f"LLM config file not found at {_LLM_CONFIG_PATH}",
            user_message="The application is missing required configuration. Contact your administrator.",
        )

    try:
        with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            message=f"Failed to parse {_LLM_CONFIG_PATH}: {exc}",
            user_message="The application's configuration is invalid. Contact your administrator.",
        ) from exc

    _yaml_cache = loaded or {}
    return _yaml_cache


def resolve_groq_config() -> ResolvedLLMConfig:
    """
    Returns the effective Groq configuration: YAML defaults from
    config/llm_config.yaml's providers.groq section, with any
    GROQ_MODEL / GROQ_TEMPERATURE / GROQ_MAX_TOKENS /
    GROQ_TIMEOUT_SECONDS environment variable overrides applied on
    top (see config.settings.AppSettings.groq_model etc. — None means
    "no override," so the YAML value is used).

    Raises:
        ConfigurationError: llm_config.yaml is missing, malformed, or
            has no `providers.groq` section.
    """
    yaml_config = _load_yaml_config()
    groq_yaml = yaml_config.get("providers", {}).get("groq")

    if groq_yaml is None:
        raise ConfigurationError(
            message="config/llm_config.yaml has no 'providers.groq' section.",
            user_message="The application is missing required configuration. Contact your administrator.",
        )

    settings = get_settings()

    resolved = ResolvedLLMConfig(
        model=settings.groq_model or groq_yaml["model"],
        temperature=(
            settings.groq_temperature if settings.groq_temperature is not None else groq_yaml["temperature"]
        ),
        max_tokens=(
            settings.groq_max_tokens if settings.groq_max_tokens is not None else groq_yaml["max_tokens"]
        ),
        timeout_seconds=(
            settings.groq_timeout_seconds
            if settings.groq_timeout_seconds is not None
            else groq_yaml["timeout_seconds"]
        ),
    )

    logger.debug(
        "groq_config_resolved",
        model=resolved.model,
        temperature=resolved.temperature,
        max_tokens=resolved.max_tokens,
        timeout_seconds=resolved.timeout_seconds,
    )

    return resolved


def resolve_insight_generation_config() -> ResolvedInsightConfig:
    """
    Returns the effective config for the business-insights LLM call:
    YAML defaults from config/llm_config.yaml's top-level
    insight_generation section.

    Unlike resolve_groq_config(), this has no environment-variable
    override layer — insight generation temperature/max_tokens are
    not exposed as user-tunable env vars in V1. Add
    INSIGHT_TEMPERATURE / INSIGHT_MAX_TOKENS to config.settings if
    that becomes necessary later; keeping it YAML-only for now avoids
    speculative configurability nobody has asked for.

    Raises:
        ConfigurationError: llm_config.yaml is missing, malformed, or
            has no `insight_generation` section.
    """
    yaml_config = _load_yaml_config()
    insight_yaml = yaml_config.get("insight_generation")

    if insight_yaml is None:
        raise ConfigurationError(
            message="config/llm_config.yaml has no 'insight_generation' section.",
            user_message="The application is missing required configuration. Contact your administrator.",
        )

    resolved = ResolvedInsightConfig(
        temperature=insight_yaml["temperature"],
        max_tokens=insight_yaml["max_tokens"],
    )

    logger.debug(
        "insight_generation_config_resolved",
        temperature=resolved.temperature,
        max_tokens=resolved.max_tokens,
    )

    return resolved


def clear_cache() -> None:
    """Clears the cached YAML config. Used by tests that need to
    reload config/llm_config.yaml after modifying it, or that swap in
    a different file path."""
    global _yaml_cache
    _yaml_cache = None
