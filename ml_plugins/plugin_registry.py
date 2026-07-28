"""
ml_plugins/plugin_registry.py
================================

Discovers and exposes available BaseMLPlugin implementations, and
routes ML-intent requests to the correct one.

WHY A REGISTRY INSTEAD OF HARDCODED IF/ELIF ROUTING
-----------------------------------------------------------
Same factory-pattern reasoning used for LLM providers
(core/llm/llm_client_factory.py) and DB connectors
(db/connectors/connection_manager.py), applied a third time for
consistency: core/orchestration/conversation_manager.py should depend
on "give me the plugin for capability X," never on a growing if/elif
chain across five plugin types. Adding a 6th ML capability later means
registering one new class here — nothing in conversation_manager.py
changes.

This also gives intent_classifier.py / conversation_manager.py a
single place to ask "what ML capabilities even exist right now,"
which is needed to build the routing prompt or to validate that a
classified suggested_ml_capability is actually a real, registered
plugin before attempting to route to it.
"""

from __future__ import annotations

from ml_plugins.base_plugin import BaseMLPlugin

_registry: dict[str, BaseMLPlugin] = {}


def register_plugin(plugin: BaseMLPlugin) -> None:
    """
    Registers a plugin instance under its capability_name.

    Implementation note: in the implementation phase, plugin
    instances for each subpackage (ml_plugins/regression/,
    ml_plugins/forecasting/, etc.) will be registered here at module
    import time or via an explicit startup routine called from
    app/main.py — deferred, since no concrete plugins exist yet at
    skeleton stage.
    """
    _registry[plugin.capability_name] = plugin


def get_plugin(capability_name: str) -> BaseMLPlugin:
    """
    Returns the registered plugin for a given capability name.

    Raises:
        PluginNotFoundError: no plugin registered under that name —
            e.g. intent_classifier suggested a capability that hasn't
            been implemented yet.
    """
    if capability_name not in _registry:
        from exceptions.domain_exceptions import PluginNotFoundError

        raise PluginNotFoundError(
            f"No ML plugin registered for capability '{capability_name}'.",
            user_message=f"The '{capability_name}' capability isn't available yet.",
        )
    return _registry[capability_name]


def list_available_capabilities() -> list[str]:
    """Returns all currently registered capability names — used to
    build routing prompts and to validate user-facing 'what can you
    do' responses against what's actually implemented, not aspirational."""
    return list(_registry.keys())
