"""
db/connectors/sqlite_connector.py
====================================

SQLite implementation of BaseDBConnector. The V1 default and only
supported dialect; PostgreSQL and MySQL connectors follow the same
interface in Phase 2 (see base_connector.py).

WHAT THIS FILE IS RESPONSIBLE FOR
------------------------------------
  1. Translating a SQLAlchemy connection URL into a working Engine for
     SQLite specifically, including its quirks (file existence
     semantics, read-only URI mode, threading defaults).
  2. Validating that the target file actually exists BEFORE attempting
     a connection, so failures are reported as a clear, specific
     "file not found at <path>" rather than a generic SQLAlchemy
     OperationalError surfacing from deep inside a connection attempt.
  3. Enforcing read-only access at the OS/SQLite level when
     `read_only=True`, per the defense-in-depth model described in
     config/security_rules.yaml — this is layer #2, independent of
     whatever the SQL validator (layer #1) does.

NOTE ON read_only FOR SQLITE SPECIFICALLY
--------------------------------------------
SQLite doesn't have database-level roles/permissions like Postgres or
MySQL. True read-only enforcement is achieved via sqlite3's URI
connection mode ("file:<path>?mode=ro", passed with uri=True), which
opens the underlying file in read-only mode at the SQLite engine
layer — an actual write attempt against a connection opened this way
raises sqlite3.OperationalError: "attempt to write a readonly
database". This has been verified directly (not assumed): opening a
populated database via this mode allows SELECT and raises on INSERT.
This is what makes it a genuine second line of defense, independent
of whatever core/nl2sql/sql_validator.py (layer #1) already rejected.

HOW READ-ONLY MODE IS WIRED INTO SQLALCHEMY
------------------------------------------------
SQLAlchemy's create_engine() with a plain "sqlite:///path" URL does
NOT open the file read-only — passing mode=ro requires calling
Python's own sqlite3.connect() with a "file:" URI and uri=True, which
is not expressible through SQLAlchemy's normal DSN string for SQLite.
The supported mechanism for this is SQLAlchemy's `creator` parameter:
a zero-argument callable that returns an already-open DBAPI
connection, which SQLAlchemy then wraps and pools normally. This
connector uses `creator` specifically for the read_only=True path;
for read_only=False, the ordinary "sqlite:///path" URL is used and
SQLAlchemy manages connection creation itself.

WHY FILE EXISTENCE IS CHECKED HERE, NOT LEFT TO SQLALCHEMY/SQLITE
-------------------------------------------------------------------------
SQLite's default behavior is to silently CREATE a new, empty database
file if the target path doesn't exist and the connection isn't opened
read-only. For this application, that default is actively dangerous:
a user with a mistyped path wouldn't get an error — they'd get a
successful connection to a brand-new, empty database, and would only
discover the problem several steps later when schema introspection
finds zero tables, with no obvious explanation why. Checking existence
explicitly, before any connection attempt, converts that into an
immediate, specific, actionable error
(exceptions.domain_exceptions.DatabaseFileNotFoundError).

When read_only=True, sqlite3's URI mode=ro already refuses to open a
missing file on its own (verified: raises OperationalError "unable to
open database file"), so the explicit pre-check is technically
redundant on that path — it is kept anyway because it produces an
earlier, clearer, more specific exception type and message than
letting that failure surface from inside the DBAPI connect call.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from db.connectors.base_connector import BaseDBConnector
from exceptions.domain_exceptions import (
    DatabaseConnectionError,
    DatabaseFileNotFoundError,
)
from logging_setup.logger import LogCategory, LogFields, get_logger

logger = get_logger(__name__)


class SQLiteConnector(BaseDBConnector):
    """SQLite implementation of BaseDBConnector. See module docstring
    for the read-only enforcement and file-existence-check design."""

    @property
    def dialect_name(self) -> str:
        return "sqlite"

    def create_engine(
        self,
        connection_url: str,
        read_only: bool,
        **kwargs: object,
    ) -> Engine:
        """
        Creates a SQLAlchemy Engine for a SQLite database file.

        Args:
            connection_url: SQLAlchemy-format SQLite URL, e.g.
                "sqlite:///./data/sample.db" (relative) or
                "sqlite:////absolute/path/to/sample.db" (absolute), or
                "sqlite:///:memory:" for an in-memory database (tests
                only — never used in normal operation).
            read_only: if True, the engine is built using sqlite3's
                URI 'mode=ro' connect mode (see module docstring for
                why this is the only reliable mechanism for SQLite,
                and verification that it genuinely blocks writes).
                Should be True for all query-execution paths, per
                settings.enforce_read_only_role.
            **kwargs: accepts `allow_create_if_missing: bool` (default
                False). If True AND read_only is False, permits
                SQLite's default "create the file if it doesn't
                exist" behavior. Mirrors
                settings.sqlite_allow_create_if_missing; accepted as a
                kwarg rather than read from settings directly inside
                this method, so this connector stays usable with an
                explicitly-provided connection_url (e.g. from the
                Settings page testing a new path) without depending on
                the global settings singleton. Ignored when
                read_only=True, since a read-only connection can never
                create a file regardless of this flag. Declared via
                **kwargs (not a named parameter) to satisfy
                BaseDBConnector's shared interface signature — see
                that class's docstring on why dialect-specific options
                are passed this way.

        Returns:
            A configured SQLAlchemy Engine, with connectivity already
            verified once during creation (see _verify_connectivity)
            so malformed URLs or permission errors surface immediately
            as DatabaseConnectionError, not as a confusing failure
            several layers away in query_executor.py later.

        Raises:
            DatabaseFileNotFoundError: the target file does not exist
                and (read_only is True) OR (allow_create_if_missing is
                False). Checked BEFORE any DBAPI connection attempt.
            DatabaseConnectionError: the connection URL is malformed,
                or engine/connection creation otherwise fails for a
                reason other than a missing file.
        """
        allow_create_if_missing = bool(kwargs.get("allow_create_if_missing", False))
        start = time.monotonic()
        db_path = self._extract_file_path(connection_url)
        is_memory_db = db_path is None

        log = logger.bind(
            category=LogCategory.DB_CONNECTION,
            **{LogFields.DB_DIALECT: self.dialect_name},
            **({LogFields.DB_PATH: str(db_path)} if db_path else {"db_path": ":memory:"}),
            read_only=read_only,
        )

        if not is_memory_db:
            self._check_file_exists_or_raise(
                db_path, read_only, allow_create_if_missing, log
            )

        try:
            if is_memory_db:
                engine = self._create_memory_engine()
            elif read_only:
                engine = self._create_read_only_engine(db_path)
            else:
                engine = self._create_read_write_engine(connection_url)

            self._verify_connectivity(engine)

        except SQLAlchemyError as exc:
            log.error("db_engine_creation_failed", error=str(exc))
            raise DatabaseConnectionError(
                message=f"Failed to create SQLAlchemy engine for '{connection_url}': {exc}",
                user_message="Couldn't connect to the database. Check your connection settings.",
            ) from exc
        except sqlite3.Error as exc:
            log.error("db_engine_creation_failed", error=str(exc))
            raise DatabaseConnectionError(
                message=f"SQLite error opening '{connection_url}' (read_only={read_only}): {exc}",
                user_message="Couldn't connect to the database. Check your connection settings.",
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        log.info("db_engine_created", **{LogFields.LATENCY_MS: round(elapsed_ms, 2)})

        return engine

    def test_connection(self, engine: Engine) -> bool:
        """
        Performs a lightweight connectivity check by opening a
        connection and running `SELECT 1`.

        Used by the Streamlit Settings page (app/pages/4_settings.py)
        to validate a connection string before saving it.

        Returns:
            True if the connection succeeds. Does not raise on
            failure — returns False and logs the reason, since this
            is meant to be used as a UI-friendly boolean check.
            Callers that need the specific failure reason should use
            create_engine() directly and inspect the raised exception.
        """
        log = logger.bind(category=LogCategory.DB_CONNECTION, dialect=self.dialect_name)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("db_connection_test_succeeded")
            return True
        except SQLAlchemyError as exc:
            log.warning("db_connection_test_failed", error=str(exc))
            return False

    # --- internal helpers ----------------------------------------------

    @staticmethod
    def _extract_file_path(connection_url: str) -> Path | None:
        """
        Extracts the filesystem path from a SQLAlchemy SQLite URL.

        Returns None for in-memory databases (":memory:" or an empty
        path component).

        Handles both 3-slash (relative) and 4-slash (absolute) forms:
            sqlite:///relative/path.db   -> relative/path.db
            sqlite:////abs/path.db        -> /abs/path.db

        Raises:
            DatabaseConnectionError: the URL doesn't start with the
                expected "sqlite:///" scheme.
        """
        prefix = "sqlite:///"
        if not connection_url.startswith(prefix):
            raise DatabaseConnectionError(
                message=f"Unsupported SQLite connection URL format: '{connection_url}'",
                user_message="The database connection string isn't in a valid format.",
            )

        remainder = connection_url[len(prefix):]

        if remainder in ("", ":memory:"):
            return None

        # 4-slash form (sqlite:////abs/path.db) leaves remainder
        # starting with "/", which Path() already treats as absolute —
        # no special-casing needed beyond the prefix strip above.
        return Path(remainder)

    @staticmethod
    def _check_file_exists_or_raise(
        db_path: Path,
        read_only: bool,
        allow_create_if_missing: bool,
        log: object,
    ) -> None:
        """Raises DatabaseFileNotFoundError if db_path doesn't exist
        and creation isn't permitted in this context. Creates the
        parent directory (not the file itself) when creation IS
        permitted, since SQLite creates the file but not intermediate
        directories on its own."""
        if db_path.is_file():
            return

        if read_only or not allow_create_if_missing:
            log.error("db_connection_failed_file_not_found", reason="sqlite_file_missing")
            raise DatabaseFileNotFoundError(
                message=(
                    f"SQLite database file not found at path: {db_path} "
                    f"(read_only={read_only}, "
                    f"allow_create_if_missing={allow_create_if_missing})"
                ),
                user_message=(
                    f"Couldn't find a database file at '{db_path}'. "
                    "Check that the path is correct."
                ),
            )

        db_path.parent.mkdir(parents=True, exist_ok=True)
        log.warning("db_file_missing_will_create", reason="allow_create_if_missing_enabled")

    @staticmethod
    def _create_read_only_engine(db_path: Path) -> Engine:
        """
        Builds an Engine backed by a single sqlite3 connection opened
        in URI mode=ro, via SQLAlchemy's `creator` parameter — the
        verified mechanism for true SQLite read-only enforcement (see
        module docstring).

        The path is URL-quoted before being embedded in the "file:"
        URI, since paths containing characters like spaces or '?'
        would otherwise corrupt the URI's query-string parsing.

        poolclass=StaticPool: a single underlying sqlite3 connection
        is reused for the engine's lifetime rather than SQLAlchemy
        attempting to open additional pooled connections via the same
        creator callable concurrently. This avoids subtle pool-sizing
        complexity for what is, for a read-only single-file SQLite
        database in this application's V1 usage pattern (one
        connected database per session), an unnecessary degree of
        concurrency to support. Revisit if/when concurrent multi-query
        execution within a single session becomes a real requirement.
        """
        quoted_path = quote(str(db_path))
        uri = f"file:{quoted_path}?mode=ro"

        def _creator() -> sqlite3.Connection:
            return sqlite3.connect(uri, uri=True, check_same_thread=False)

        return sa_create_engine("sqlite://", creator=_creator, poolclass=StaticPool)

    @staticmethod
    def _create_read_write_engine(connection_url: str) -> Engine:
        """
        Builds a normal read-write Engine using SQLAlchemy's standard
        SQLite URL handling.

        check_same_thread=False: Streamlit's execution model can hand
        requests to different threads across reruns; SQLite's default
        same-thread check would raise spuriously in that environment.
        This is acceptable here because concurrency safety for this
        application comes from one engine per session
        (app/state/session_state.py's st.cache_resource wrapping), not
        from SQLite's own thread check.
        """
        return sa_create_engine(connection_url, connect_args={"check_same_thread": False})

    @staticmethod
    def _create_memory_engine() -> Engine:
        """
        Builds an Engine for an in-memory SQLite database, used only
        in tests.

        StaticPool is required here: in-memory SQLite databases are
        scoped to the single DBAPI connection that created them.
        Without StaticPool, each new connection pulled from
        SQLAlchemy's default pool would see a separate, empty
        in-memory database, since "memory" isn't shared across
        connections the way a file on disk is.
        """
        return sa_create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

    @staticmethod
    def _verify_connectivity(engine: Engine) -> None:
        """
        Forces an eager connection + trivial query right after engine
        creation, so failures (bad permissions, corrupt file, etc.)
        surface immediately as part of create_engine() rather than
        lazily on the first real query several layers away.
        """
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
