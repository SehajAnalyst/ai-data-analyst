"""
app/state/session_state.py
=============================

The only module that reads/writes st.session_state directly.
Wraps it with typed accessors and applies st.cache_resource to
expensive objects (database engine, schema) so they are created
once per Streamlit server process, not on every rerun.

WHY THIS EXISTS
----------------
Streamlit reruns the entire script on every user interaction.
Without this layer, session_state key strings get scattered across
every page and component — a typo is a silent runtime bug. This
module centralises all state access behind named functions with
clear types.

CACHING RULES
--------------
st.cache_resource  — objects that are expensive to create and are
                     safe to share across all users/sessions
                     (database engine, schema). These live for the
                     entire server process lifetime.
st.session_state   — objects that are per-user and per-session
                     (chat history, query history, current db path).
                     Lost on browser refresh.

DO NOT put business logic here. This module only manages state
storage and retrieval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import streamlit as st

from core.memory.conversation_manager import ConversationManager

# ── State key constants ────────────────────────────────────────────────────
# Defined here as module-level constants so a typo is a NameError (loud)
# not a KeyError that silently creates a new key (quiet and wrong).

_SESSION_ID = "session_id"
_CHAT_HISTORY = "chat_history"
_CONVERSATION_MANAGER = "conversation_manager"
_DB_URL = "db_url"
_DB_CONNECTED = "db_connected"
_SCHEMA_CACHE_KEY = "schema_cache_key"
_CURRENT_CONVERSATION_ID = "current_conversation_id"

# ── Per-turn data structures ───────────────────────────────────────────────
# NOTE: QueryHistoryItem previously lived here and served the sidebar's
# "Recent Queries" display. It has been retired in favor of
# core.memory.conversation_manager.ConversationTurn, which now serves
# BOTH the sidebar display and LLM follow-up context — one canonical
# structure instead of two overlapping ones. See
# core/memory/conversation_manager.py's module docstring for the full
# rationale.


@dataclass
class ChatMessage:
    """One message in the chat thread."""

    role: str                    # "user" or "assistant"
    content: str                 # display text
    timestamp: datetime = field(default_factory=datetime.now)
    # Optional rich payload attached to assistant messages:
    sql: str | None = None
    validation_error: str | None = None
    query_result: Any = None     # QueryResult dataclass or None
    error: str | None = None
    insight: Any = None   
    chart_metadata: dict | None = None# InsightResult dataclass or None — optional,
                                  # absent if generation failed or was skipped


# ── Initialisation ─────────────────────────────────────────────────────────


def init_session_state() -> None:
    """
    Initialises all session_state keys to their defaults if not already
    present. Call once at the top of every page before accessing state.

    Idempotent: safe to call on every Streamlit rerun — only absent
    keys are written, existing values are never overwritten.
    """
    defaults: dict[str, Any] = {
    _SESSION_ID: str(uuid.uuid4()),
    _CHAT_HISTORY: [],
    _DB_URL: None,
    _DB_CONNECTED: False,
    _SCHEMA_CACHE_KEY: None,
    _CURRENT_CONVERSATION_ID: None,   # <-- Add this
}
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Constructed separately from the defaults dict above: each session
    # needs its OWN ConversationManager instance, not a value copied
    # from a shared dict literal — the same reason a mutable default
    # argument in a function signature is a classic bug (all callers
    # would share one instance). Streamlit's rerun model means this
    # check runs on every rerun, but the `not in` guard makes it a
    # no-op after the first.
    if _CONVERSATION_MANAGER not in st.session_state:
        st.session_state[_CONVERSATION_MANAGER] = ConversationManager()


# ── Session ID ─────────────────────────────────────────────────────────────


def get_session_id() -> str:
    return st.session_state[_SESSION_ID]


# ── Chat history ───────────────────────────────────────────────────────────


def get_chat_history() -> list[ChatMessage]:
    return st.session_state[_CHAT_HISTORY]


def add_chat_message(message: ChatMessage) -> None:
    st.session_state[_CHAT_HISTORY].append(message)


def clear_chat_history() -> None:
    st.session_state[_CHAT_HISTORY] = []


# ── Conversation memory ─────────────────────────────────────────────────────


def get_conversation_manager() -> ConversationManager:
    """
    Returns this session's ConversationManager — the canonical store
    for turn history, used both by the sidebar's "Recent Queries"
    display and by core.memory.context_builder for LLM follow-up
    context. One instance per session (see init_session_state above),
    not st.cache_resource — conversation memory is per-user, not
    shareable across the process like the DB engine/schema are.
    """
    return st.session_state[_CONVERSATION_MANAGER]


def clear_chat_and_memory() -> None:
    """
    "Clear Chat": full reset. Wipes the visible transcript
    (ChatMessage list) AND the conversation memory used for follow-up
    resolution. Use this when the user wants to start completely
    fresh.
    """
    clear_chat_history()
    get_conversation_manager().clear_all()


def clear_context_only() -> None:
    """
    "Clear Context": wipes only the conversation memory used for
    follow-up resolution. The visible chat transcript is left
    untouched — the user can still see what was asked, but the AI
    will no longer use those prior turns to interpret a new question
    as a follow-up. Useful for pivoting to an unrelated topic without
    losing the visible conversation on screen.
    """
    get_conversation_manager().clear_context()


# ── Database connection state ──────────────────────────────────────────────


def get_db_url() -> str | None:
    return st.session_state[_DB_URL]


def set_db_url(url: str) -> None:
    st.session_state[_DB_URL] = url


def is_db_connected() -> bool:
    return st.session_state[_DB_CONNECTED]


def set_db_connected(connected: bool) -> None:
    st.session_state[_DB_CONNECTED] = connected


# ── Cached expensive resources ─────────────────────────────────────────────
# st.cache_resource caches at the server-process level — one engine
# for all sessions. This is correct for a single-database app. For
# multi-user multi-database deployments, the cache key must include
# the database URL. The _by_url variants below handle that.


@st.cache_resource(show_spinner=False)
def _get_engine_cached(db_url: str):
    """Create and cache one SQLAlchemy engine per unique db_url."""
    from db.connectors.connection_manager import get_engine
    return get_engine(connection_url=db_url, read_only=True)


@st.cache_resource(show_spinner=False)
def _get_schema_cached(db_url: str):
    """Introspect and cache schema once per unique db_url."""
    from core.schema.schema_introspector import introspect_schema
    engine = _get_engine_cached(db_url)
    return introspect_schema(engine, include_row_counts=True)


def get_engine(db_url: str | None = None):
    """
    Returns the cached SQLAlchemy engine. If db_url is None, reads
    from session state (the currently connected database).
    Raises if no database is configured.
    """
    url = db_url or get_db_url()
    if not url:
        from exceptions.domain_exceptions import DatabaseConnectionError
        raise DatabaseConnectionError(
            message="No database URL configured.",
            user_message="No database is connected. Configure one in the sidebar.",
        )
    return _get_engine_cached(url)


def get_schema(db_url: str | None = None):
    """Returns the cached DatabaseSchema for the connected database."""
    url = db_url or get_db_url()
    if not url:
        from exceptions.domain_exceptions import SchemaIntrospectionError
        raise SchemaIntrospectionError(
            message="No database URL configured.",
            user_message="No database is connected. Configure one in the sidebar.",
        )
    return _get_schema_cached(url)


def invalidate_schema_cache() -> None:
    """
    Forces schema re-introspection on next access. Clears Streamlit's
    cache_resource for _get_schema_cached so a fresh call will re-run
    the introspection against the database.
    """
    _get_schema_cached.clear()

# ── Current Conversation ID ─────────────────────────────────────────────

def get_current_conversation_id() -> str | None:
    return st.session_state[_CURRENT_CONVERSATION_ID]


def set_current_conversation_id(
    conversation_id: str | None,
) -> None:
    st.session_state[_CURRENT_CONVERSATION_ID] = conversation_id

def has_active_conversation() -> bool:
    return get_current_conversation_id() is not None


def clear_current_conversation() -> None:
    set_current_conversation_id(None)