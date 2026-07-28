"""
core/schema/schema_context_builder.py
========================================

Selects which subset of the full DatabaseSchema to inject into the LLM
prompt for a given question, and formats it as prompt-ready text.

V1 IMPLEMENTATION: FULL-SCHEMA INCLUSION
------------------------------------------
For V1, this module uses the simple path: include all tables in every
prompt. This is correct for small databases (which is what V1 targets
with SQLite as the default) and incorrect for large ones. The
architectural boundary between "deciding what to include" (this module)
and "generating SQL" (core/nl2sql/sql_generator.py) exists precisely
so that switching from full-inclusion to RAG-based retrieval in V2
happens HERE and nowhere else — sql_generator.py doesn't need to
change, and callers don't need to change. Their interface
(build_schema_context() -> SchemaContext) stays identical.

The threshold for when RAG becomes necessary is roughly 20-25 tables.
Above that, prompt token cost and LLM accuracy both start degrading.
The settings.schema_rag_enabled flag and settings.schema_rag_top_k_tables
are already in place (config/settings.py) — the V2 RAG path branches
on that flag inside this function, once implemented.

WHY THIS IS ITS OWN MODULE, SEPARATE FROM sql_generator.py
------------------------------------------------------------
These are different concerns that change for different reasons:
- "What tables are relevant to this question" (this module) changes
  based on schema size, RAG model quality, and retrieval tuning.
- "How to generate SQL given the relevant schema" (sql_generator.py)
  changes based on prompt engineering, model choice, retry logic.
Mixing them would require touching SQL generation code every time the
schema-selection strategy is tuned or upgraded.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_settings
from core.schema.schema_introspector import (
    DatabaseSchema,
    TableInfo,
    build_schema_summary,
)
from logging_setup.logger import LogCategory, LogFields, get_logger

logger = get_logger(__name__)


@dataclass
class SchemaContext:
    """
    The final, prompt-ready schema context for one question.
    Returned to core/nl2sql/sql_generator.py for prompt assembly.

    Fields:
        relevant_tables: the TableInfo objects for every table included
            in formatted_text. sql_validator.py (once implemented)
            uses this list to check that the LLM's generated SQL only
            references tables it was actually told about.
        formatted_text: the text block injected into the LLM prompt
            where {schema_context} appears in the template.
        retrieval_method: "full_schema" in V1; "rag_retrieval" once
            the RAG path is implemented. Logged per question so
            latency and accuracy differences between the two modes
            are attributable in the logs.
        table_names: set of table names in relevant_tables — used by
            sql_validator.py for fast existence checks without
            iterating the full list every time.
        all_column_names_by_table: mapping of table_name -> set of
            column names, for the same validator use case.
    """

    relevant_tables: list[TableInfo]
    formatted_text: str
    retrieval_method: str

    @property
    def table_names(self) -> set[str]:
        return {t.name for t in self.relevant_tables}

    @property
    def all_column_names_by_table(self) -> dict[str, set[str]]:
        return {t.name: {c.name for c in t.columns} for t in self.relevant_tables}


def build_schema_context(
    schema: DatabaseSchema,
    user_question: str,
    conversation_history_summary: str | None = None,
) -> SchemaContext:
    """
    Returns the schema context to inject into the SQL generation
    prompt for this specific question.

    Args:
        schema: full introspection result from schema_introspector,
            typically from schema_cache.get_cached_schema() — this
            function does not re-introspect.
        user_question: the current question. Logged and, in the future
            RAG path, used as the retrieval query.
        conversation_history_summary: optional compact summary of
            recent turns. In the RAG path (V2), prior context affects
            which tables are retrieved — "now break it down by region"
            doesn't name any table, so retrieval has to consider what
            tables were relevant in the preceding turn. Accepted as a
            parameter here (even in V1's full-inclusion path, where
            it's unused) so callers don't need to change their call
            signature when V2 adds the RAG path.

    Returns:
        SchemaContext with all tables from `schema` included (V1).

    Raises:
        SchemaIntrospectionError: propagated from schema if the schema
            object itself is malformed (should not normally occur if
            schema was built by schema_introspector.introspect_schema).
    """
    settings = get_settings()

    if settings.schema_rag_enabled and len(schema.tables) > 0:
        # V2 placeholder — when RAG is implemented, this branch will:
        #   1. Embed user_question + conversation_history_summary
        #   2. Query the vector store for top-k relevant table embeddings
        #   3. Return only those tables in relevant_tables / formatted_text
        # For now, fall through to full-schema inclusion with a log warning.
        logger.warning(
            "schema_rag_requested_but_not_implemented",
            category=LogCategory.SCHEMA_DISCOVERY,
            question_preview=user_question[:100],
        )

    formatted_text = build_schema_summary(schema)
    retrieval_method = "full_schema"

    logger.debug(
        "schema_context_built",
        category=LogCategory.SCHEMA_DISCOVERY,
        retrieval_method=retrieval_method,
        **{
            LogFields.TABLE_COUNT: len(schema.tables),
        },
    )

    return SchemaContext(
        relevant_tables=schema.tables,
        formatted_text=formatted_text,
        retrieval_method=retrieval_method,
    )
