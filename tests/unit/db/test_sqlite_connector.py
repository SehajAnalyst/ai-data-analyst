"""
tests/unit/db/test_sqlite_connector.py
=========================================

Unit tests for db/connectors/sqlite_connector.py and
db/connectors/connection_manager.py.

WHY THESE TESTS USE REAL SQLITE FILES (tmp_path), NOT MOCKS
-------------------------------------------------------------------
The entire point of this module is correct interaction with SQLite's
actual file-system and read-only semantics — mocking sqlite3/SQLAlchemy
here would test that our code calls the mocks correctly, not that the
read-only enforcement and file-existence checks actually work. pytest's
`tmp_path` fixture gives each test an isolated real directory, so these
tests exercise the real behavior this module exists to guarantee.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from db.connectors.connection_manager import get_connector_for_dialect, get_engine
from db.connectors.sqlite_connector import SQLiteConnector
from config.settings import DatabaseDialect
from exceptions.domain_exceptions import (
    DatabaseConnectionError,
    DatabaseFileNotFoundError,
    UnsupportedDialectError,
)


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    """Creates a real SQLite file with one table and one row."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'widget')")
    conn.commit()
    conn.close()
    return db_path


class TestSQLiteConnectorCreateEngine:
    def test_connects_to_existing_file_read_only(self, populated_db: Path) -> None:
        connector = SQLiteConnector()
        engine = connector.create_engine(f"sqlite:///{populated_db}", read_only=True)

        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM items")).fetchall()

        assert rows == [(1, "widget")]

    def test_read_only_blocks_writes(self, populated_db: Path) -> None:
        connector = SQLiteConnector()
        engine = connector.create_engine(f"sqlite:///{populated_db}", read_only=True)

        with pytest.raises(OperationalError, match="readonly"):
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO items VALUES (2, 'gadget')"))
                conn.commit()

    def test_read_write_allows_writes(self, populated_db: Path) -> None:
        connector = SQLiteConnector()
        engine = connector.create_engine(f"sqlite:///{populated_db}", read_only=False)

        with engine.connect() as conn:
            conn.execute(text("INSERT INTO items VALUES (2, 'gadget')"))
            conn.commit()
            rows = conn.execute(text("SELECT * FROM items")).fetchall()

        assert len(rows) == 2

    def test_missing_file_raises_specific_error_read_only(self, tmp_path: Path) -> None:
        connector = SQLiteConnector()
        missing_path = tmp_path / "nope.db"

        with pytest.raises(DatabaseFileNotFoundError) as exc_info:
            connector.create_engine(f"sqlite:///{missing_path}", read_only=True)

        assert str(missing_path) in exc_info.value.message
        assert "Couldn't find" in exc_info.value.user_message

    def test_missing_file_raises_when_create_not_allowed(self, tmp_path: Path) -> None:
        connector = SQLiteConnector()
        missing_path = tmp_path / "nope.db"

        with pytest.raises(DatabaseFileNotFoundError):
            connector.create_engine(
                f"sqlite:///{missing_path}",
                read_only=False,
                allow_create_if_missing=False,
            )

    def test_missing_file_creates_when_explicitly_allowed(self, tmp_path: Path) -> None:
        connector = SQLiteConnector()
        new_path = tmp_path / "new.db"
        assert not new_path.exists()

        connector.create_engine(
            f"sqlite:///{new_path}",
            read_only=False,
            allow_create_if_missing=True,
        )

        assert new_path.exists()

    def test_missing_file_never_created_when_read_only_even_if_allowed(
        self, tmp_path: Path
    ) -> None:
        """read_only=True must take precedence over
        allow_create_if_missing=True — a read-only request should
        never result in file creation, regardless of that flag."""
        connector = SQLiteConnector()
        missing_path = tmp_path / "nope.db"

        with pytest.raises(DatabaseFileNotFoundError):
            connector.create_engine(
                f"sqlite:///{missing_path}",
                read_only=True,
                allow_create_if_missing=True,
            )

        assert not missing_path.exists()

    def test_in_memory_database_works(self) -> None:
        connector = SQLiteConnector()
        engine = connector.create_engine("sqlite:///:memory:", read_only=False)

        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE t (x INTEGER)"))
            conn.execute(text("INSERT INTO t VALUES (1)"))
            conn.commit()
            rows = conn.execute(text("SELECT * FROM t")).fetchall()

        assert rows == [(1,)]

    def test_malformed_url_raises_connection_error(self) -> None:
        connector = SQLiteConnector()

        with pytest.raises(DatabaseConnectionError):
            connector.create_engine("not-a-valid-sqlite-url", read_only=True)


class TestSQLiteConnectorTestConnection:
    def test_returns_true_for_valid_engine(self, populated_db: Path) -> None:
        connector = SQLiteConnector()
        engine = connector.create_engine(f"sqlite:///{populated_db}", read_only=True)

        assert connector.test_connection(engine) is True


class TestConnectionManager:
    def test_get_engine_uses_explicit_url_override(self, populated_db: Path) -> None:
        engine = get_engine(connection_url=f"sqlite:///{populated_db}", read_only=True)

        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM items")).fetchall()

        assert rows == [(1, "widget")]

    def test_get_connector_for_dialect_returns_sqlite_connector(self) -> None:
        connector = get_connector_for_dialect(DatabaseDialect.SQLITE)
        assert connector.dialect_name == "sqlite"

    def test_get_connector_for_unregistered_dialect_raises(self) -> None:
        with pytest.raises(UnsupportedDialectError):
            get_connector_for_dialect(DatabaseDialect.POSTGRESQL)
