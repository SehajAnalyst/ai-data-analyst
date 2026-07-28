"""
scripts/example_schema_discovery_usage.py
============================================

Runnable example demonstrating the schema discovery module
(core/schema/schema_introspector.py), chained with the database
connection module (db/connectors/) built previously.

    PYTHONPATH=. python3 scripts/example_schema_discovery_usage.py

This mirrors how upstream consumers will use this module later:
  - core/schema/schema_context_builder.py calls introspect_schema()
    (via core/schema/schema_cache.py for the cached path) to get a
    DatabaseSchema, then selects a relevant subset per question.
  - core/nl2sql/sql_generator.py calls build_schema_summary() (on
    that relevant subset) to get prompt-ready text.
  - app/pages/2_schema_explorer.py calls introspect_schema() directly
    to render the full schema for browsing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from db.connectors.connection_manager import get_engine
from core.schema.schema_introspector import build_schema_summary, introspect_schema
from exceptions.domain_exceptions import SchemaIntrospectionError


def _create_demo_database(path: Path) -> None:
    """Creates a throwaway SQLite file with related tables, so this
    example has real foreign keys and relationships to discover."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TEXT,
            amount REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_name TEXT,
            quantity INTEGER,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );
        """
    )
    conn.execute("INSERT INTO customers VALUES (1, 'Acme Corp', 'West')")
    conn.execute("INSERT INTO customers VALUES (2, 'Globex', 'East')")
    conn.execute("INSERT INTO orders VALUES (1, 1, '2026-01-15', 250.00)")
    conn.execute("INSERT INTO orders VALUES (2, 2, '2026-02-03', 89.99)")
    conn.execute("INSERT INTO order_items VALUES (1, 1, 'Widget', 3)")
    conn.commit()
    conn.close()


def example_structured_schema(db_path: Path) -> None:
    """
    Pattern 1: get the structured DatabaseSchema and inspect it
    programmatically — what schema_context_builder.py and
    sql_validator.py (table/column existence checks) will do.
    """
    print("\n--- Pattern 1: structured schema, programmatic access ---")

    engine = get_engine(connection_url=f"sqlite:///{db_path}", read_only=True)
    schema = introspect_schema(engine, include_row_counts=True)

    print(f"Tables found: {schema.table_names}")

    orders_table = schema.get_table("orders")
    print(f"\n'orders' table columns:")
    for col in orders_table.columns:
        flags = []
        if col.is_primary_key:
            flags.append("PK")
        if col.is_foreign_key:
            flags.append(f"FK->{col.foreign_key_target}")
        print(f"  {col.name} ({col.data_type}){' [' + ', '.join(flags) + ']' if flags else ''}")

    print(f"\nRelationships in the database:")
    for rel in schema.relationships:
        print(f"  {rel.as_sentence()}")


def example_json_serialization(db_path: Path) -> None:
    """
    Pattern 2: serialize the structured schema to JSON — useful for
    caching to disk, or for any future API layer that needs to expose
    the schema as a response payload.
    """
    print("\n--- Pattern 2: JSON-serializable structured output ---")

    engine = get_engine(connection_url=f"sqlite:///{db_path}", read_only=True)
    schema = introspect_schema(engine, include_row_counts=True)

    # dataclasses.asdict() works directly since every model in
    # schema_introspector.py is a plain dataclass with no
    # SQLAlchemy objects embedded — this is exactly why the module
    # docstring insists on NOT leaking raw SQLAlchemy Table/Column
    # objects into the output shape.
    schema_dict = asdict(schema)
    print(json.dumps(schema_dict, indent=2)[:1500] + "\n  ... (truncated for display)")


def example_llm_ready_summary(db_path: Path) -> None:
    """
    Pattern 3: the human-readable summary that gets embedded directly
    into the SQL generation prompt — what
    core/nl2sql/sql_generator.py will actually send to the LLM.
    """
    print("\n--- Pattern 3: human-readable summary for LLM prompts ---")

    engine = get_engine(connection_url=f"sqlite:///{db_path}", read_only=True)
    schema = introspect_schema(engine, include_row_counts=True, include_sample_values=True)

    summary = build_schema_summary(schema)
    print(summary)


def example_empty_database() -> None:
    """
    Pattern 4: how callers handle an empty database — the Schema
    Explorer page will catch nothing here (default raise_on_empty=False)
    and just render schema.is_empty as an empty state.
    """
    print("\n--- Pattern 4: empty database handling ---")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        empty_path = Path(f.name)
    sqlite3.connect(empty_path).close()  # valid empty SQLite file

    engine = get_engine(connection_url=f"sqlite:///{empty_path}", read_only=True)
    schema = introspect_schema(engine)  # raise_on_empty=False (default)

    print(f"is_empty: {schema.is_empty}")
    print(f"Summary shown to user: {build_schema_summary(schema)!r}")

    empty_path.unlink()


def example_error_handling() -> None:
    """
    Pattern 5: error handling for a corrupted/inaccessible database —
    the pattern app/pages/2_schema_explorer.py will use to show a
    clean error state instead of crashing.
    """
    print("\n--- Pattern 5: error handling for corrupted database ---")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        corrupt_path = Path(f.name)
        f.write(b"not a real sqlite file")

    try:
        engine = get_engine(connection_url=f"sqlite:///{corrupt_path}", read_only=True)
        introspect_schema(engine)
    except SchemaIntrospectionError as exc:
        print(f"[shown to user]: {exc.user_message}")
        print(f"[logged internally]: {exc.message}")
    finally:
        corrupt_path.unlink()


if __name__ == "__main__":
    demo_db_path = Path("./data/_example_schema_demo.db")
    _create_demo_database(demo_db_path)

    example_structured_schema(demo_db_path)
    example_json_serialization(demo_db_path)
    example_llm_ready_summary(demo_db_path)
    example_empty_database()
    example_error_handling()

    demo_db_path.unlink(missing_ok=True)
    print("\nAll examples completed; demo database cleaned up.")
