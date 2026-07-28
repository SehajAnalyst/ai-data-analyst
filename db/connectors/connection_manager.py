"""
db/connectors/connection_manager.py
======================================

Picks the correct BaseDBConnector implementation based on
settings.db_dialect, and exposes a single get_engine() entry point for
the rest of the application.

WHY THIS SITS ABOVE the individual connectors
------------------------------------------------
This is the dialect-selection factory — same pattern as
core/llm/llm_client_factory.py, applied to databases instead of LLMs.
Code elsewhere (schema introspector, query executor) calls:

    from db.connectors.connection_manager import get_engine

    engine = get_engine()

...and never needs to know or care whether that engine is backed by
SQLite, Postgres, or MySQL. For V1, only SQLite is registered — adding
Postgres/MySQL in Phase 2 means adding their connector classes
(already stubbed, see postgres_connector.py / mysql_connector.py) and
one line each in _CONNECTORS below. Nothing else in the codebase
changes.

ENGINE CACHING
---------------
This module deliberately does NOT cache engines itself (no
module-level dict of "engine per connection string"). Engine lifetime
caching is the Streamlit layer's responsibility
(app/state/session_state.py, via st.cache_resource) — see that
module's docstring. Caching here as well would create two competing
cache layers with unclear invalidation semantics (which one wins when
a user changes their connection string?). Keeping this module a plain,
uncached factory function keeps the caching policy in exactly one
place.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from config.settings import DatabaseDialect, get_settings
from db.connectors.base_connector import BaseDBConnector
from db.connectors.sqlite_connector import SQLiteConnector
from exceptions.domain_exceptions import UnsupportedDialectError
from logging_setup.logger import LogCategory, get_logger

logger = get_logger(__name__)

# Dispatch table: DatabaseDialect -> connector instance.
# Connectors are stateless (all state is passed per-call as
# arguments), so one shared instance per dialect is safe and avoids
# pointless re-instantiation on every get_engine() call.
#
# PostgresConnector / MySQLConnector are intentionally NOT registered
# yet (Phase 2) even though their classes already exist as stubs —
# registering an unimplemented connector here would make
# UnsupportedDialectError fire later and more confusingly (inside the
# connector's own NotImplementedError) instead of immediately and
# clearly at dispatch time.
_CONNECTORS: dict[DatabaseDialect, BaseDBConnector] = {
    DatabaseDialect.SQLITE: SQLiteConnector(),
}


def get_engine(
    connection_url: str | None = None,
    read_only: bool = True,
) -> Engine:
    """
    Returns a configured SQLAlchemy Engine for the application's
    configured (or explicitly passed) database dialect.

    Args:
        connection_url: overrides settings.database_url if provided
            (used by the Streamlit settings page when a user is
            testing a new connection before saving it).
        read_only: passed through to the connector; should be True
            for all query-execution paths. False is reserved for
            schema introspection internals only, if ever needed, and
            must never be flipped to False casually — see
            sqlite_connector.py's module docstring on why read-only
            enforcement is a required safety layer, not optional
            hardening.

    Returns:
        A connected, verified SQLAlchemy Engine (see
        SQLiteConnector._verify_connectivity — connectivity is checked
        eagerly during creation, not left fully lazy).

    Raises:
        UnsupportedDialectError: settings.db_dialect has no registered
            connector (e.g. db_dialect is set to postgresql/mysql
            before Phase 2's connectors are registered above).
        DatabaseFileNotFoundError: (SQLite) the target file doesn't
            exist and creation isn't permitted in this context.
        DatabaseConnectionError: connection/engine creation failed for
            any other reason.
    """
    settings = get_settings()
    dialect = settings.db_dialect
    url = connection_url or settings.database_url

    connector = _CONNECTORS.get(dialect)
    if connector is None:
        logger.error(
            "db_unsupported_dialect",
            category=LogCategory.DB_CONNECTION,
            dialect=dialect.value,
        )
        raise UnsupportedDialectError(
            message=f"No registered connector for dialect '{dialect.value}'.",
            user_message=(
                f"'{dialect.value}' databases aren't supported yet. "
                "SQLite is currently the only supported database type."
            ),
        )

    # SQLite-specific kwarg, passed through **kwargs per
    # BaseDBConnector's shared signature (see that class's docstring).
    # Harmless to pass for any dialect since unused kwargs are simply
    # ignored by connectors that don't read them — but currently only
    # SQLiteConnector is registered, so this is the only path exercised.
    extra_kwargs: dict[str, object] = {}
    if dialect == DatabaseDialect.SQLITE:
        extra_kwargs["allow_create_if_missing"] = settings.sqlite_allow_create_if_missing

    return connector.create_engine(url, read_only=read_only, **extra_kwargs)


def get_connector_for_dialect(dialect: DatabaseDialect) -> BaseDBConnector:
    """
    Returns the registered connector instance for a given dialect,
    without creating an engine. Used by callers that need
    dialect-specific behavior beyond engine creation — e.g.
    test_connection() from the Settings page, or
    core.schema.schema_introspector needing to know the dialect's name
    for prompt construction.

    Raises:
        UnsupportedDialectError: no connector registered for this dialect.
    """
    connector = _CONNECTORS.get(dialect)
    if connector is None:
        raise UnsupportedDialectError(
            message=f"No registered connector for dialect '{dialect.value}'.",
            user_message=(
                f"'{dialect.value}' databases aren't supported yet. "
                "SQLite is currently the only supported database type."
            ),
        )
    return connector
