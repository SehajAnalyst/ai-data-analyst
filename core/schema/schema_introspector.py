"""
core/schema/schema_introspector.py
=====================================

Reads structural metadata (tables, columns, types, primary keys,
foreign keys, and table-to-table relationships) from the user's
connected database using SQLAlchemy's reflection capabilities
(`sqlalchemy.inspect`), and produces both a structured DatabaseSchema
object and a human-readable text summary suitable for an LLM prompt.

WHY THIS IS ITS OWN MODULE, SEPARATE FROM schema_context_builder.py
-------------------------------------------------------------------------
Introspection ("what does the database structurally contain") and
context building ("what subset of that should I show the LLM for
THIS question") are different concerns with different failure modes
and different caching needs:

  - Introspection is expensive-ish (DB round trip) but changes rarely
    — cache aggressively, invalidate manually or on a long TTL (see
    core/schema/schema_cache.py).
  - Context building is cheap (works off the cached introspection
    result) but runs on every single question.

Keeping them separate means schema_context_builder.py can be unit
tested against a fixed, fake DatabaseSchema without touching a real
database at all.

OUTPUT SHAPE
------------
This module's output (DatabaseSchema and its nested dataclasses) is a
plain, serializable data structure — not raw SQLAlchemy Table/Column
objects — because the introspection result needs to be:
  - cached (potentially to disk, for the future schema RAG index)
  - JSON-serialized into LLM prompts
  - independent of SQLAlchemy internals leaking into core/llm prompt code

WHAT "TABLE RELATIONSHIPS" MEANS HERE, DISTINCT FROM "FOREIGN KEYS"
-------------------------------------------------------------------------
Foreign keys (ColumnInfo.is_foreign_key /
ColumnInfo.foreign_key_target) are a per-COLUMN fact: "this column
references that column." A relationship
(TableRelationship, on DatabaseSchema.relationships) is a per-TABLE-PAIR
fact, derived FROM the foreign keys but represented at the right level
for two different consumers:
  - The LLM prompt (sql_generation) needs "orders relates to customers
    via orders.customer_id -> customers.id" as a single readable
    statement, not scattered per-column annotations the model has to
    reassemble itself.
  - Future JOIN-suggestion logic (e.g. "the user asked for X and Y,
    which live in different tables — what's the join path") needs to
    walk table-to-table edges, which is awkward to do from
    column-level FK data alone without first building this view.

This module derives relationships from the same FK reflection data
used for ColumnInfo — there is no separate, independent detection
mechanism, since SQLite's reflected foreign key constraints ARE the
authoritative relationship signal. "Detecting" a relationship beyond
declared FKs (e.g. inferring one from a column naming convention like
`customer_id` with no actual FK constraint) is NOT attempted here:
guessing relationships that aren't actually enforced by the schema
would silently feed wrong join assumptions into SQL generation later,
which is a correctness risk, not a convenience worth the false-positive
rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from exceptions.domain_exceptions import EmptyDatabaseError, SchemaIntrospectionError
from logging_setup.logger import LogCategory, LogFields, get_logger

logger = get_logger(__name__)


# --- Data models -----------------------------------------------------------


@dataclass
class ColumnInfo:
    """One column's structural metadata."""

    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    foreign_key_target: str | None = None  # "other_table.column" if applicable


@dataclass
class TableRelationship:
    """
    One table-to-table relationship, derived from a foreign key
    constraint. See module docstring for why this is represented
    separately from the per-column FK facts on ColumnInfo.

    Directional: `from_table` is the table holding the foreign key
    column; `to_table` is the table being referenced. "orders
    references customers" is from_table=orders, to_table=customers —
    not symmetric, since the FK constraint itself isn't symmetric.
    """

    from_table: str
    from_column: str
    to_table: str
    to_column: str

    def as_sentence(self) -> str:
        """Human-readable rendering used in the LLM prompt summary —
        kept as a method here (not duplicated string formatting in
        the summary builder) so there's exactly one place this
        phrasing is defined."""
        return f"{self.from_table}.{self.from_column} references {self.to_table}.{self.to_column}"


@dataclass
class TableInfo:
    """One table's structural metadata."""

    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count_estimate: int | None = None
    # Optional: sample values per column, used to give the LLM concrete
    # examples (e.g. knowing `status` contains 'active'/'inactive'
    # rather than guessing) — populated only when explicitly requested,
    # since pulling sample data has its own cost and privacy
    # considerations (see introspect_schema()'s include_sample_values
    # parameter docstring).
    sample_values: dict[str, list[str]] = field(default_factory=dict)

    def get_column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def primary_key_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_primary_key]


@dataclass
class DatabaseSchema:
    """Full introspection result for one connected database."""

    dialect: str
    tables: list[TableInfo] = field(default_factory=list)
    relationships: list[TableRelationship] = field(default_factory=list)

    def get_table(self, name: str) -> TableInfo | None:
        return next((t for t in self.tables if t.name == name), None)

    @property
    def is_empty(self) -> bool:
        return len(self.tables) == 0

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def relationships_for_table(self, table_name: str) -> list[TableRelationship]:
        """All relationships touching a given table, in either
        direction. Used by schema_context_builder.py when assembling
        prompt context for a question that involves a specific table —
        knowing what it joins to (or is joined from) is part of what
        makes that context useful."""
        return [
            r
            for r in self.relationships
            if r.from_table == table_name or r.to_table == table_name
        ]


# --- Introspection -----------------------------------------------------------


def introspect_schema(
    engine: Engine,
    include_sample_values: bool = False,
    include_row_counts: bool = True,
    raise_on_empty: bool = False,
) -> DatabaseSchema:
    """
    Reflects the full schema of the database behind `engine` and
    returns it as a DatabaseSchema.

    Args:
        engine: SQLAlchemy engine for the target database (from
            db/connectors/connection_manager.py). Should be a
            read_only=True engine — introspection only ever reads
            metadata and small samples, never writes, so there's no
            reason to use a read-write engine for this call.
        include_sample_values: if True, pulls a small number of
            distinct sample values per column. Off by default — this
            issues additional queries per column and, more
            importantly, means actual user data values get sent to
            the LLM provider as part of prompt context later. That's a
            privacy/data-residency decision the caller (and
            ultimately the end user, via a settings toggle) should
            make explicitly, not a silent default.
        include_row_counts: if True, runs a COUNT(*) per table to
            populate TableInfo.row_count_estimate. Default True since
            this is cheap for typical table sizes and genuinely useful
            in the human-readable summary (helps the LLM judge
            plausible query results, e.g. "don't expect millions of
            rows"). Can be disabled for very large schemas where even
            COUNT(*) per table adds up.
        raise_on_empty: if True, raises EmptyDatabaseError when the
            database has zero tables, instead of returning an empty
            DatabaseSchema. Default False because "empty database" is
            a normal, displayable state (see EmptyDatabaseError's
            docstring) — most callers (Schema Explorer page) want to
            render an empty-state message, not handle an exception.
            Set True for callers where proceeding with zero tables
            genuinely doesn't make sense (e.g. a startup sanity check
            that wants to fail loudly).

    Returns:
        DatabaseSchema with tables, columns, and relationships
        populated. If the database has no tables, returns a
        DatabaseSchema with an empty `tables` list (NOT an error)
        unless raise_on_empty=True — see EmptyDatabaseError docstring
        on why this is a separate, deliberate condition from
        SchemaIntrospectionError.

    Raises:
        SchemaIntrospectionError: reflection failed for a genuine
            reason (corrupt database file, permission denied,
            connection dropped mid-reflection).
        EmptyDatabaseError: database has zero tables AND
            raise_on_empty=True.
    """
    start = time.monotonic()
    log = logger.bind(category=LogCategory.SCHEMA_DISCOVERY, dialect=engine.dialect.name)

    try:
        inspector = sa_inspect(engine)
        table_names = inspector.get_table_names()
    except SQLAlchemyError as exc:
        log.error("schema_introspection_failed", error=str(exc))
        raise SchemaIntrospectionError(
            message=f"Failed to read table list from database: {exc}",
            user_message="Couldn't read the database structure. The database may be corrupted or inaccessible.",
        ) from exc

    if not table_names:
        log.warning("schema_introspection_empty_database")
        if raise_on_empty:
            raise EmptyDatabaseError(
                message="Connected database contains zero tables.",
                user_message="This database doesn't have any tables yet.",
            )
        return DatabaseSchema(dialect=engine.dialect.name, tables=[], relationships=[])

    try:
        tables = [
            _introspect_table(
                inspector, engine, table_name, include_sample_values, include_row_counts, log
            )
            for table_name in table_names
        ]
        relationships = _derive_relationships(inspector, table_names, log)
    except SQLAlchemyError as exc:
        log.error("schema_introspection_failed", error=str(exc))
        raise SchemaIntrospectionError(
            message=f"Failed to read table metadata: {exc}",
            user_message="Couldn't read the database structure. The database may be corrupted or inaccessible.",
        ) from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    log.info(
        "schema_introspection_succeeded",
        **{
            LogFields.TABLE_COUNT: len(tables),
            LogFields.RELATIONSHIP_COUNT: len(relationships),
            LogFields.LATENCY_MS: round(elapsed_ms, 2),
        },
    )

    return DatabaseSchema(dialect=engine.dialect.name, tables=tables, relationships=relationships)


def _introspect_table(
    inspector: object,
    engine: Engine,
    table_name: str,
    include_sample_values: bool,
    include_row_counts: bool,
    log: object,
) -> TableInfo:
    """
    Builds a TableInfo for one table: columns (with types, nullability,
    PK/FK flags), optional row count, optional sample values.

    KNOWN SQLITE QUIRK HANDLED HERE: a column declared
    `INTEGER PRIMARY KEY` in SQLite is a rowid alias and is always
    functionally NOT NULL, even though SQLAlchemy's reflection reports
    `nullable=True` for it (verified directly against a real SQLite
    table — this is not a hypothetical edge case). Reporting it as
    nullable=True would be actively misleading in both the structured
    schema and the LLM-facing summary ("this column can be NULL" is
    false for a primary key). This function corrects that: any column
    that is part of the primary key is reported as nullable=False,
    regardless of what raw reflection says.
    """
    raw_columns = inspector.get_columns(table_name)
    pk_constraint = inspector.get_pk_constraint(table_name)
    pk_columns = set(pk_constraint.get("constrained_columns") or [])

    fk_constraints = inspector.get_foreign_keys(table_name)
    # Map: column_name -> "referred_table.referred_column", built once
    # per table rather than re-scanning fk_constraints per column.
    fk_targets: dict[str, str] = {}
    for fk in fk_constraints:
        referred_table = fk.get("referred_table")
        constrained_cols = fk.get("constrained_columns") or []
        referred_cols = fk.get("referred_columns") or []
        for local_col, remote_col in zip(constrained_cols, referred_cols):
            fk_targets[local_col] = f"{referred_table}.{remote_col}"

    columns: list[ColumnInfo] = []
    for raw_col in raw_columns:
        col_name = raw_col["name"]
        is_pk = col_name in pk_columns
        is_fk = col_name in fk_targets

        # See docstring: primary key columns are never actually
        # nullable in practice, regardless of raw reflection output.
        nullable = bool(raw_col["nullable"]) and not is_pk

        columns.append(
            ColumnInfo(
                name=col_name,
                data_type=str(raw_col["type"]),
                nullable=nullable,
                is_primary_key=is_pk,
                is_foreign_key=is_fk,
                foreign_key_target=fk_targets.get(col_name),
            )
        )

    row_count_estimate: int | None = None
    if include_row_counts:
        row_count_estimate = _safe_row_count(engine, table_name, log)

    sample_values: dict[str, list[str]] = {}
    if include_sample_values:
        sample_values = _fetch_sample_values(engine, table_name, columns, log)

    return TableInfo(
        name=table_name,
        columns=columns,
        row_count_estimate=row_count_estimate,
        sample_values=sample_values,
    )


def _derive_relationships(
    inspector: object,
    table_names: list[str],
    log: object,
) -> list[TableRelationship]:
    """
    Builds the table-to-table relationship list from each table's
    reflected foreign key constraints. See module docstring for why
    this is derived strictly from declared FKs, with no inference
    beyond that.
    """
    relationships: list[TableRelationship] = []
    for table_name in table_names:
        fk_constraints = inspector.get_foreign_keys(table_name)
        for fk in fk_constraints:
            referred_table = fk.get("referred_table")
            constrained_cols = fk.get("constrained_columns") or []
            referred_cols = fk.get("referred_columns") or []
            for local_col, remote_col in zip(constrained_cols, referred_cols):
                relationships.append(
                    TableRelationship(
                        from_table=table_name,
                        from_column=local_col,
                        to_table=referred_table,
                        to_column=remote_col,
                    )
                )
    return relationships


def _safe_row_count(engine: Engine, table_name: str, log: object) -> int | None:
    """
    Runs SELECT COUNT(*) for one table. Returns None (rather than
    raising) on failure, since a row-count failure for one table
    shouldn't abort introspection of the entire schema — this is a
    nice-to-have annotation, not structural data the rest of the
    pipeline depends on.

    Table name is identifier-quoted via SQLAlchemy's dialect-aware
    quoting (engine.dialect.identifier_preparer), not raw
    string-formatted into the query, since table names are not
    user-controlled SQL injection input in this context but should
    still be handled correctly for names containing spaces or
    reserved words.
    """
    try:
        quoted_name = engine.dialect.identifier_preparer.quote(table_name)
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {quoted_name}"))
            count = result.scalar()
            return int(count) if count is not None else None
    except SQLAlchemyError as exc:
        log.warning("row_count_failed", table_name=table_name, error=str(exc))
        return None


def _fetch_sample_values(
    engine: Engine,
    table_name: str,
    columns: list[ColumnInfo],
    log: object,
    max_samples_per_column: int = 5,
) -> dict[str, list[str]]:
    """
    Pulls up to `max_samples_per_column` distinct values per column.
    Best-effort: a failure on one column logs a warning and is
    skipped, rather than aborting the whole table's sample collection.

    NOTE: this sends real user data values into whatever calls this
    with include_sample_values=True — see introspect_schema()'s
    docstring on why this is opt-in, not a default.
    """
    samples: dict[str, list[str]] = {}
    quoted_table = engine.dialect.identifier_preparer.quote(table_name)

    for col in columns:
        try:
            quoted_col = engine.dialect.identifier_preparer.quote(col.name)
            query = text(
                f"SELECT DISTINCT {quoted_col} FROM {quoted_table} "
                f"WHERE {quoted_col} IS NOT NULL LIMIT :limit"
            )
            with engine.connect() as conn:
                result = conn.execute(query, {"limit": max_samples_per_column})
                values = [str(row[0]) for row in result.fetchall()]
            if values:
                samples[col.name] = values
        except SQLAlchemyError as exc:
            log.warning("sample_value_fetch_failed", table_name=table_name, column=col.name, error=str(exc))
            continue

    return samples


# --- Human-readable summary for LLM prompts ---------------------------------


def build_schema_summary(schema: DatabaseSchema) -> str:
    """
    Renders a DatabaseSchema as a compact, human-readable text block
    suitable for inclusion in an LLM prompt (core/nl2sql/sql_generator.py).

    WHY A SEPARATE FUNCTION, NOT A __str__/__repr__ ON DatabaseSchema
    -------------------------------------------------------------------------
    DatabaseSchema is a plain data model — its job is to hold
    structured, queryable data (used by schema_context_builder.py's
    table-selection logic, by core/nl2sql/sql_validator.py's
    table/column existence checks, etc). Prompt formatting is a
    presentation concern with its own evolving requirements (token
    budget tuning, what to include/omit per model), and conflating it
    with the data model would mean every prompt-wording tweak touches
    the same file as schema correctness logic. Keeping this separate
    also makes it easy to have MULTIPLE summary formats later (e.g. a
    terser one for small-context models) without touching
    DatabaseSchema at all.

    FORMAT NOTES
    ------------
    Primary keys are marked inline (PK). Foreign keys are marked
    inline (FK -> target) AND repeated as a separate "Relationships"
    section in plain-English sentences — the inline marker helps the
    model see, while reading a single table's columns, which column is
    the join key; the separate relationships section gives it the
    join PATH between tables without having to mentally cross-reference
    every table's column list. Both views are kept deliberately small
    and non-redundant in wording (the inline marker is terse; the
    relationships section is the one place the full sentence form
    appears).

    Row counts are included per table when available, since they help
    the model calibrate expectations (e.g. avoid assuming a 5-row
    lookup table needs a LIMIT, or recognize a fact table is large
    enough that aggregation is the sensible approach).
    """
    if schema.is_empty:
        return "This database has no tables."

    lines: list[str] = [f"Database type: {schema.dialect}", ""]

    for table in schema.tables:
        row_info = f" (~{table.row_count_estimate} rows)" if table.row_count_estimate is not None else ""
        lines.append(f"Table: {table.name}{row_info}")

        for col in table.columns:
            markers: list[str] = []
            if col.is_primary_key:
                markers.append("PK")
            if col.is_foreign_key and col.foreign_key_target:
                markers.append(f"FK -> {col.foreign_key_target}")
            if not col.nullable:
                markers.append("NOT NULL")

            marker_str = f" [{', '.join(markers)}]" if markers else ""
            lines.append(f"  - {col.name}: {col.data_type}{marker_str}")

            if col.name in table.sample_values:
                examples = ", ".join(table.sample_values[col.name])
                lines.append(f"      example values: {examples}")

        lines.append("")  # blank line between tables

    if schema.relationships:
        lines.append("Relationships:")
        for rel in schema.relationships:
            lines.append(f"  - {rel.as_sentence()}")

    return "\n".join(lines)
