"""
app/pages/2_schema_explorer.py
=================================

Lets users browse the connected database's tables/columns directly —
serves two purposes per the architecture doc (section 6): helping
users know what's queryable, and doubling as a debugging tool during
development.

RESPONSIBILITIES (UI ONLY)
----------------------------
  - Call core.schema.schema_introspector.introspect_schema() (via
    core.schema.schema_cache for the cached path) and render the
    result as an explorable tree/table.
  - Provide a "Refresh Schema" action that calls
    core.schema.schema_cache.invalidate() — the manual invalidation
    path described in schema_cache.py's docstring.
  - For schema_rag_enabled deployments, optionally surface which
    tables a sample question would retrieve, as a debugging aid for
    tuning the RAG retrieval (genuinely useful during development of
    the retrieval logic, not just an end-user feature).

Implementation deferred to implementation phase.
"""
