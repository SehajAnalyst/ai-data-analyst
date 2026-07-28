"""
app/components/sql_viewer.py
================================

Renders generated SQL in a collapsible expander with syntax
highlighting. Collapsed by default — most users care about the
answer, not the SQL. Available for those who want to verify or learn.
"""

from __future__ import annotations

import streamlit as st


def render_sql_viewer(sql: str, validation_error: str | None = None) -> None:
    """
    Renders an expander containing the SQL query.

    Args:
        sql: the generated (and validated) SQL to display.
        validation_error: if set, the SQL failed validation and this
            message explains why. The SQL is shown anyway so the user
            can understand what was attempted.
    """
    if not sql:
        return

    label = "⚠ Generated SQL (validation failed)" if validation_error else "📄 Generated SQL"

    with st.expander(label, expanded=False):
        st.code(sql, language="sql")
        if validation_error:
            st.error(f"Validation error: {validation_error}")
