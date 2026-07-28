"""
scripts/example_db_connection_usage.py
=========================================

Runnable example demonstrating the database connection module
(db/connectors/). Not part of the application itself — a reference
script you can run directly to see the module working end-to-end:

    PYTHONPATH=. python3 scripts/example_db_connection_usage.py

This mirrors exactly how the module will be called by upstream
consumers later (schema discovery, query execution, Streamlit UI) —
see the module-level docstring further down for that mapping.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import text

from db.connectors.connection_manager import get_connector_for_dialect, get_engine
from config.settings import DatabaseDialect
from exceptions.domain_exceptions import (
    DatabaseConnectionError,
    DatabaseFileNotFoundError,
    UnsupportedDialectError,
)


def _create_demo_database(path: Path) -> None:
    """Creates a throwaway SQLite file with sample data, purely so
    this example script has something real to connect to."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT)")
    conn.execute("DELETE FROM customers")
    conn.execute("INSERT INTO customers VALUES (1, 'Acme Corp', 'West')")
    conn.execute("INSERT INTO customers VALUES (2, 'Globex', 'East')")
    conn.commit()
    conn.close()


def example_basic_connection(db_path: Path) -> None:
    """
    Pattern 1: connect using settings.database_url (the normal,
    production path — no explicit connection_url override).

    This is what core/schema/schema_introspector.py and
    core/execution/query_executor.py will do: call get_engine() with
    no arguments and trust it to read configuration from
    config.settings.get_settings() and return a ready, read-only engine.
    """
    print("\n--- Pattern 1: default connection from settings.database_url ---")

    # In a real run this would come from .env / environment variables.
    # Set here directly only so this example is self-contained.
    import os
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    from config.settings import get_settings
    get_settings.cache_clear()  # picks up the env var set above

    engine = get_engine()  # read_only=True by default
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM customers")).fetchall()
        print(f"Connected successfully. Rows: {rows}")


def example_explicit_connection_url(db_path: Path) -> None:
    """
    Pattern 2: connect using an explicit connection_url, overriding
    settings.database_url for this one call.

    This is what app/pages/4_settings.py will do when a user is
    testing a NEW database connection before saving it as their
    default — the override exists specifically so "test this path"
    doesn't require mutating global settings first.
    """
    print("\n--- Pattern 2: explicit connection_url override ---")

    engine = get_engine(connection_url=f"sqlite:///{db_path}", read_only=True)
    connector = get_connector_for_dialect(DatabaseDialect.SQLITE)

    is_valid = connector.test_connection(engine)
    print(f"Connection test result: {is_valid}")


def example_read_only_enforcement(db_path: Path) -> None:
    """
    Pattern 3: demonstrates that read_only=True is a REAL safety
    boundary, not just a flag name — an actual write attempt against
    a read-only-opened engine fails at the SQLite engine level.

    Relevant to core/execution/query_executor.py, which always
    requests read_only=True and relies on this as its second line of
    defense behind core/nl2sql/sql_validator.py's SQL-level checks.
    """
    print("\n--- Pattern 3: read-only enforcement is real ---")

    engine = get_engine(connection_url=f"sqlite:///{db_path}", read_only=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO customers VALUES (99, 'Should Fail', 'Nowhere')"))
            conn.commit()
        print("UNEXPECTED: write succeeded (this should not happen)")
    except Exception as exc:
        print(f"Write correctly blocked: {type(exc).__name__}")


def example_error_handling() -> None:
    """
    Pattern 4: how upstream callers (orchestrator, Streamlit pages)
    are expected to handle this module's exceptions — catching the
    specific exception types from exceptions/domain_exceptions.py and
    using their `.user_message` for display, `.message` for logs.
    """
    print("\n--- Pattern 4: error handling pattern for callers ---")

    try:
        get_engine(connection_url="sqlite:///./data/this_does_not_exist.db", read_only=True)
    except DatabaseFileNotFoundError as exc:
        # This is the pattern app/pages/4_settings.py and
        # core/orchestration/conversation_manager.py will both use:
        # show exc.user_message to the user, log exc.message internally.
        print(f"[shown to user]: {exc.user_message}")
        print(f"[logged internally]: {exc.message}")
    except (DatabaseConnectionError, UnsupportedDialectError) as exc:
        print(f"[shown to user]: {exc.user_message}")


if __name__ == "__main__":
    demo_db_path = Path("./data/_example_demo.db")
    _create_demo_database(demo_db_path)

    example_basic_connection(demo_db_path)
    example_explicit_connection_url(demo_db_path)
    example_read_only_enforcement(demo_db_path)
    example_error_handling()

    demo_db_path.unlink(missing_ok=True)
    print("\nAll examples completed; demo database cleaned up.")
