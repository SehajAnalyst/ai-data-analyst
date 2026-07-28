"""
core/schema/schema_cache.py
==============================

Caches the result of schema_introspector.introspect_schema() so
repeated questions in a session don't trigger repeated database
metadata round trips.

WHY A DEDICATED CACHE MODULE RATHER THAN @st.cache_resource INLINE
-------------------------------------------------------------------------
core/ must stay independent of Streamlit (see architecture doc and
folder-structure rationale — this is what allows a future migration
off Streamlit without rewriting the AI pipeline). If caching were done
via Streamlit decorators directly inside schema_introspector.py,
core/ would have a hard Streamlit dependency. Instead, this module
defines a plain-Python cache; the Streamlit layer (app/state/) wraps
calls to it with `st.cache_resource` at the UI boundary, where that
coupling is acceptable.

INVALIDATION
------------
Schema can legitimately change (user adds a table, migration runs).
This module supports explicit invalidation (called from the Streamlit
settings page's "Refresh Schema" button) and, as a future enhancement,
TTL-based expiry — not building TTL into V1 since manual refresh is
sufficient for a single-user/small-team tool and adds a "is my schema
stale right now" debugging question that isn't worth it yet.
"""

from __future__ import annotations

from core.schema.schema_introspector import DatabaseSchema

_cache: dict[str, DatabaseSchema] = {}


def get_cached_schema(connection_key: str) -> DatabaseSchema | None:
    """Returns the cached schema for a given connection, or None if
    not yet cached / invalidated."""
    return _cache.get(connection_key)


def set_cached_schema(connection_key: str, schema: DatabaseSchema) -> None:
    """Stores a freshly introspected schema in the cache."""
    _cache[connection_key] = schema


def invalidate(connection_key: str) -> None:
    """Forces the next access to re-introspect. Called from the
    Streamlit settings page's manual refresh action."""
    _cache.pop(connection_key, None)
