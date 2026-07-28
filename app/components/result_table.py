"""
app/components/result_table.py
==================================

Renders QueryResult as an interactive DataFrame with row count and
execution time metadata. Explicitly signals truncation — showing
"first 1,000 rows" is honest; showing a capped table silently is not.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.execution.query_executor import QueryResult
from logging_setup.logger import get_logger

logger = get_logger(__name__)


def render_result_table(result: QueryResult) -> None:
    """
    Displays the query result as a Streamlit DataFrame with metadata.

    Args:
        result: QueryResult from query_executor.execute_query().

    Guarded end-to-end (added during production hardening): building
    the DataFrame or rendering it can fail on unexpected data shapes
    (e.g. a BLOB column pandas/Streamlit can't display, ragged rows).
    That failure must not propagate to Streamlit's top level and show
    a raw traceback — the user still gets a clear, if less detailed,
    message instead. This mirrors the same discipline already applied
    in app/components/chart_renderer.py's render_visualization().
    """
    try:
        _render_result_table_inner(result)
    except Exception as exc:
        logger.warning("result_table_rendering_failed", error=str(exc)[:200])
        st.error("The results couldn't be displayed as a table, but the query ran successfully.")


def _render_result_table_inner(result: QueryResult) -> None:
    if result.row_count == 0:
        st.info("The query returned no results.")
        return

    df = pd.DataFrame(result.rows, columns=result.columns)

    # Metadata row: count, timing, truncation warning.
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        label = f"{result.row_count:,} rows"
        if result.truncated:
            label += " (truncated)"
        st.metric("Rows returned", label)
    with meta_col2:
        st.metric("Columns", len(result.columns))
    with meta_col3:
        st.metric("Execution time", f"{result.execution_time_ms:.0f}ms")

    if result.truncated:
        st.warning(
            f"Showing first {result.row_count:,} rows. "
            "The full result set was larger. Refine your question to see more specific data."
        )

    st.dataframe(df, use_container_width=True)
