"""
tests/unit/core/test_schema_introspector.py
==============================================

Unit tests for core/schema/schema_introspector.py.

WHY THESE USE REAL SQLITE DATABASES (tmp_path), NOT MOCKS
-------------------------------------------------------------------
Same rationale as tests/unit/db/test_sqlite_connector.py: the entire
point of this module is correct interaction with SQLAlchemy's
reflection against real SQLite metadata, including dialect-specific
quirks (the INTEGER PRIMARY KEY nullable quirk in particular). Mocking
the Inspector would test that our code calls the mock correctly, not
that introspection produces correct results against real SQLite
behavior.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from core.schema.schema_introspector import (
    build_schema_summary,
    introspect_schema,
)
from exceptions.domain_exceptions import EmptyDatabaseError, SchemaIntrospectionError


@pytest.fixture
def empty_db_engine(tmp_path: Path) -> Engine:
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    conn.close()  # creates an empty, valid SQLite file with zero tables
    return create_engine(f"sqlite:///{db_path}")


@pytest.fixture
def simple_db_engine(tmp_path: Path) -> Engine:
    """One table, no foreign keys — the simplest non-trivial case."""
    db_path = tmp_path / "simple.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, region TEXT)"
    )
    conn.execute("INSERT INTO customers VALUES (1, 'Acme Corp', 'West')")
    conn.execute("INSERT INTO customers VALUES (2, 'Globex', 'East')")
    conn.commit()
    conn.close()
    return create_engine(f"sqlite:///{db_path}")


@pytest.fixture
def related_db_engine(tmp_path: Path) -> Engine:
    """Two tables with a foreign key relationship between them."""
    db_path = tmp_path / "related.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            amount REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        """
    )
    conn.commit()
    conn.close()
    return create_engine(f"sqlite:///{db_path}")


@pytest.fixture
def corrupt_db_engine(tmp_path: Path) -> Engine:
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"this is not a valid sqlite file")
    return create_engine(f"sqlite:///{db_path}")


class TestIntrospectSchemaBasics:
    def test_reads_table_names(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        assert schema.table_names == ["customers"]

    def test_reads_column_names_and_types(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        table = schema.get_table("customers")
        assert table is not None

        col_names = [c.name for c in table.columns]
        assert col_names == ["id", "name", "region"]

        col_types = {c.name: c.data_type for c in table.columns}
        assert col_types["id"] == "INTEGER"
        assert col_types["name"] == "TEXT"

    def test_detects_primary_key(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        table = schema.get_table("customers")
        assert table.primary_key_columns == ["id"]
        assert table.get_column("id").is_primary_key is True
        assert table.get_column("name").is_primary_key is False

    def test_sqlite_integer_primary_key_corrected_to_not_nullable(
        self, simple_db_engine: Engine
    ) -> None:
        """Known SQLAlchemy/SQLite quirk: raw reflection reports
        nullable=True for INTEGER PRIMARY KEY columns even though
        they're functionally NOT NULL. This must be corrected, not
        passed through verbatim — see schema_introspector.py's
        _introspect_table docstring."""
        schema = introspect_schema(simple_db_engine)
        table = schema.get_table("customers")
        assert table.get_column("id").nullable is False

    def test_declared_not_null_column_is_not_nullable(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        table = schema.get_table("customers")
        assert table.get_column("name").nullable is False

    def test_nullable_column_is_nullable(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        table = schema.get_table("customers")
        assert table.get_column("region").nullable is True

    def test_row_count_included_by_default(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        table = schema.get_table("customers")
        assert table.row_count_estimate == 2

    def test_row_count_excluded_when_disabled(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine, include_row_counts=False)
        table = schema.get_table("customers")
        assert table.row_count_estimate is None

    def test_dialect_recorded(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        assert schema.dialect == "sqlite"


class TestForeignKeysAndRelationships:
    def test_detects_foreign_key_on_column(self, related_db_engine: Engine) -> None:
        schema = introspect_schema(related_db_engine)
        orders = schema.get_table("orders")
        customer_id_col = orders.get_column("customer_id")

        assert customer_id_col.is_foreign_key is True
        assert customer_id_col.foreign_key_target == "customers.id"

    def test_non_fk_column_not_flagged(self, related_db_engine: Engine) -> None:
        schema = introspect_schema(related_db_engine)
        orders = schema.get_table("orders")
        assert orders.get_column("amount").is_foreign_key is False
        assert orders.get_column("amount").foreign_key_target is None

    def test_relationship_derived_correctly(self, related_db_engine: Engine) -> None:
        schema = introspect_schema(related_db_engine)
        assert len(schema.relationships) == 1

        rel = schema.relationships[0]
        assert rel.from_table == "orders"
        assert rel.from_column == "customer_id"
        assert rel.to_table == "customers"
        assert rel.to_column == "id"

    def test_relationship_sentence_format(self, related_db_engine: Engine) -> None:
        schema = introspect_schema(related_db_engine)
        rel = schema.relationships[0]
        assert rel.as_sentence() == "orders.customer_id references customers.id"

    def test_relationships_for_table_finds_both_directions(
        self, related_db_engine: Engine
    ) -> None:
        schema = introspect_schema(related_db_engine)

        assert len(schema.relationships_for_table("orders")) == 1
        assert len(schema.relationships_for_table("customers")) == 1  # referenced side too

    def test_table_with_no_relationships_returns_empty(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        assert schema.relationships_for_table("customers") == []
        assert schema.relationships == []


class TestSampleValues:
    def test_sample_values_off_by_default(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        table = schema.get_table("customers")
        assert table.sample_values == {}

    def test_sample_values_populated_when_requested(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine, include_sample_values=True)
        table = schema.get_table("customers")

        assert "name" in table.sample_values
        assert set(table.sample_values["name"]) == {"Acme Corp", "Globex"}


class TestEmptyDatabase:
    def test_empty_database_returns_empty_schema_by_default(
        self, empty_db_engine: Engine
    ) -> None:
        schema = introspect_schema(empty_db_engine)
        assert schema.is_empty is True
        assert schema.tables == []

    def test_empty_database_raises_when_requested(self, empty_db_engine: Engine) -> None:
        with pytest.raises(EmptyDatabaseError):
            introspect_schema(empty_db_engine, raise_on_empty=True)

    def test_empty_database_summary_text(self, empty_db_engine: Engine) -> None:
        schema = introspect_schema(empty_db_engine)
        summary = build_schema_summary(schema)
        assert summary == "This database has no tables."


class TestErrorHandling:
    def test_corrupt_database_raises_introspection_error(
        self, corrupt_db_engine: Engine
    ) -> None:
        with pytest.raises(SchemaIntrospectionError) as exc_info:
            introspect_schema(corrupt_db_engine)

        assert exc_info.value.user_message  # has a safe, non-empty user-facing message


class TestSchemaSummary:
    def test_summary_includes_table_and_column_names(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        summary = build_schema_summary(schema)

        assert "Table: customers" in summary
        assert "id: INTEGER" in summary
        assert "name: TEXT" in summary

    def test_summary_marks_primary_key(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine)
        summary = build_schema_summary(schema)
        assert "PK" in summary

    def test_summary_includes_relationships_section(self, related_db_engine: Engine) -> None:
        schema = introspect_schema(related_db_engine)
        summary = build_schema_summary(schema)

        assert "Relationships:" in summary
        assert "orders.customer_id references customers.id" in summary

    def test_summary_omits_relationships_section_when_none_exist(
        self, simple_db_engine: Engine
    ) -> None:
        schema = introspect_schema(simple_db_engine)
        summary = build_schema_summary(schema)
        assert "Relationships:" not in summary

    def test_summary_includes_row_counts(self, simple_db_engine: Engine) -> None:
        schema = introspect_schema(simple_db_engine, include_row_counts=True)
        summary = build_schema_summary(schema)
        assert "~2 rows" in summary
