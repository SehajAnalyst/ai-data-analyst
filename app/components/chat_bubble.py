"""
app/components/chat_bubble.py
================================

Renders one complete chat turn: text, SQL expander, result table,
chart, and — when present — the business insights panel.
"""

from __future__ import annotations

import streamlit as st

from app.components.chart_renderer import render_visualization
from app.components.result_table import render_result_table
from app.components.sql_viewer import render_sql_viewer
from app.state.session_state import ChatMessage
from logging_setup.logger import get_logger

logger = get_logger(__name__)


def render_chat_message(message: ChatMessage) -> None:
    """
    Renders one message. Guarded end-to-end (added during production
    hardening): this is the single boundary every rendering call for
    a message goes through, called once per message from app/main.py.
    A failure anywhere in the rendering chain (SQL viewer, result
    table, chart, insights) must not propagate to Streamlit's top
    level — one broken message should degrade to a plain error line,
    not take down the whole page. Individual components
    (chart_renderer.py, result_table.py) already guard their own
    risky operations internally; this outer boundary is a second,
    consistent line of defense for anything those don't anticipate.
    """
    with st.chat_message(message.role):
        try:
            _render_chat_message_inner(message)
        except Exception as exc:
            logger.warning("chat_message_rendering_failed", error=str(exc)[:200], role=message.role)
            st.error("This message couldn't be fully displayed.")

def _render_chat_message_inner(message: ChatMessage):
    print(">>> INSIDE _render_chat_message_inner <<<")
    print("Role:", message.role)

    if message.role == "user":
        print("Rendering user message")
        st.markdown(message.content)
        return

    print("Rendering assistant message")

    if message.error:
        print("Rendering error")
        st.error(message.content)
        with st.expander("Error details"):
            st.code(message.error)
        return

    if message.content:
        print("Rendering content")
        st.markdown(message.content)

    if message.sql:
        print("Rendering SQL")
        render_sql_viewer(
            sql=message.sql,
            validation_error=message.validation_error,
        )

    if message.query_result is not None:
        print("Rendering table")
        render_result_table(message.query_result)

        print("Rendering chart")
        render_visualization(message.query_result)

        print("Rendering insight")
        print("INSIGHT OBJECT:", message.insight)
        _render_insight(message.insight)
        
    elif message.validation_error and not message.sql:
        print("Rendering validation warning")
        st.warning(f"Could not generate a safe query: {message.validation_error}")

def _render_insight(insight) -> None:
    """
    Renders the InsightResult panel. Absent (None) means insight
    generation was skipped or failed — nothing is shown in that case,
    since the query results and chart are already visible and
    complete without it.
    """
    if insight is None:
        return

    # Empty-result insight is a single line, not a full panel — the
    # "no results" message already appeared from render_result_table.
    if insight.is_empty:
        return

    has_content = (
        insight.summary
        or insight.key_trends
        or insight.outliers
        or insight.important_metrics
        or insight.follow_up_questions
    )
    if not has_content:
        return

    with st.expander("💡 Insights", expanded=True):
        if insight.summary:
            st.markdown(insight.summary)

        if insight.important_metrics:
            st.markdown("**Key metrics**")
            for metric in insight.important_metrics:
                st.markdown(f"- {metric}")

        if insight.key_trends:
            st.markdown("**Trends**")
            for trend in insight.key_trends:
                st.markdown(f"- {trend}")

        if insight.outliers:
            st.markdown("**Outliers**")
            for outlier in insight.outliers:
                st.markdown(f"- {outlier}")

        if insight.follow_up_questions:
            st.markdown("**You might also ask:**")
            for question in insight.follow_up_questions:
                st.markdown(f"- {question}")
