"""
app/pages/3_query_history.py
===============================

Shows past turns from the current session (and, once
db/repository/conversation_repository.py is implemented, optionally
past sessions too) with the ability to re-run a past query.

RESPONSIBILITIES (UI ONLY)
----------------------------
  - Read turns from st.session_state (current session) and/or
    db.repository.conversation_repository (persisted history, once
    implemented).
  - Re-render each past turn using the SAME app/components/ rendering
    functions used in app/pages/1_chat.py (sql_viewer, result_table,
    chart_renderer) — reused, not reimplemented, per the rationale in
    1_chat.py.
  - "Re-run" action re-executes the stored SQL directly via
    core.execution.query_executor (data may have changed since the
    original run) rather than re-invoking the full NL2SQL pipeline,
    since the SQL is already known and validated.

Implementation deferred to implementation phase.
"""
