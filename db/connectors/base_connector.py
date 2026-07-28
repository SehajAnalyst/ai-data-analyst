"""
db/connectors/base_connector.py
==================================

Abstract interface every database dialect connector
(SQLite/PostgreSQL/MySQL) must implement.

WHY THIS EXISTS
----------------
Same Strategy-pattern reasoning as core/llm/base_provider.py: the rest
of the app (schema introspection, query execution) should depend on
this interface, never on a specific dialect's connection mechanics.
That's what makes "SQLite now, Postgres/MySQL later" an additive
change instead of a rewrite.

This wraps SQLAlchemy Engine creation specifically — it does NOT wrap
query execution (that's core/execution/query_executor.py's job, which
takes a connection from here and applies safety/timeout logic on top).
Keeping "how do I connect" separate from "how do I safely execute" is
deliberate: connection concerns (pooling, dialect quirks, read-only
role setup) and execution-safety concerns (timeouts, row limits,
validation) are different responsibilities and change for different
reasons.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sqlalchemy.engine import Engine


class BaseDBConnector(ABC):
    """
    Every concrete connector (SQLiteConnector, PostgresConnector,
    MySQLConnector) extends this.
    """

    @abstractmethod
    def create_engine(self, connection_url: str, read_only: bool, **kwargs: object) -> Engine:
        """
        Returns a configured SQLAlchemy Engine for this dialect.

        Args:
            connection_url: SQLAlchemy-format connection string.
            read_only: if True, the engine MUST be configured to
                connect using read-only credentials/role where the
                dialect supports it (e.g. a dedicated read-only
                Postgres role). This is layer #2 of the defense-in-depth
                safety model described in config/security_rules.yaml —
                required, not optional, when
                settings.enforce_read_only_role is True.
            **kwargs: dialect-specific options that don't apply
                universally (e.g. SQLite's allow_create_if_missing).
                Kept as **kwargs at the interface level rather than
                forcing every dialect to share one fixed parameter
                list — Postgres/MySQL connectors (Phase 2) will accept
                and ignore SQLite-only kwargs, or define their own
                (e.g. ssl_mode), without changing this shared
                signature.

        Raises:
            DatabaseConnectionError: connection could not be established.
        """
        raise NotImplementedError

    @abstractmethod
    def test_connection(self, engine: Engine) -> bool:
        """Lightweight connectivity check, used by the Streamlit
        settings page to validate a connection string before saving it."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dialect_name(self) -> str:
        """Short identifier, e.g. 'sqlite', used in logging and in
        prompts (the LLM needs to know which SQL dialect to target —
        e.g. SQLite vs PostgreSQL have different string/date functions)."""
        raise NotImplementedError
