"""
app/pages/4_settings.py
==========================

User-configurable settings: LLM provider selection, database
connection management.

RESPONSIBILITIES (UI ONLY)
----------------------------
  - LLM provider picker, persisted to st.session_state (NOT to
    config.settings, which is process-wide and env-driven — a
    per-session override needs to flow as an explicit argument into
    core.llm.llm_client_factory.get_llm_provider(provider=...), not
    by mutating global settings, since Streamlit can serve multiple
    concurrent user sessions per process).
  - Database connection form, using
    db.connectors.base_connector.test_connection() to validate before
    saving (see that interface's docstring).
  - Manual schema refresh trigger, delegating to
    core.schema.schema_cache.invalidate() (same action also offered
    on the schema explorer page — fine for it to live in both places
    since it's a thin call either way).

SECURITY NOTE: this page should never display API key VALUES back to
the user once entered (write-only fields), even though it reads
whether a key is configured (boolean presence check) — avoids
accidental key disclosure via screen-share, screenshots, etc.

Implementation deferred to implementation phase.
"""
