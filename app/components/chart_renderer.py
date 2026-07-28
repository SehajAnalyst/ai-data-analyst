"""
app/components/chart_renderer.py
====================================

The only file in the project that both imports Plotly builders AND
calls Streamlit rendering functions. All other visualization code is
Streamlit-free; this is where those two worlds meet.

Entry point: render_visualization(result: QueryResult) -> None
Called from app/components/chat_bubble.py after a successful query.
"""

from __future__ import annotations

import streamlit as st

from core.execution.query_executor import QueryResult
from core.execution.result_analyzer import analyze_result
from core.visualization.chart_selector import ChartType, select_chart


def render_visualization(result: QueryResult) -> None:
    """
    Full pipeline: QueryResult → shape → selection → build → render.
    Silently skips rendering when no chart is appropriate (NONE type).
    Never raises — a visualization failure must not crash the chat page.

    Args:
        result: QueryResult from query_executor. If the result has
                zero rows, this is a no-op.
    """
    if result.row_count == 0:
        return

    try:
        shape = analyze_result(result)
        selection = select_chart(shape)

        if selection.chart_type == ChartType.NONE:
            return

        st.caption(f"📊 {selection.description}")

        if selection.chart_type == ChartType.BAR:
            from core.visualization.chart_builders.bar_chart import build_bar_chart
            fig = build_bar_chart(selection)
            st.plotly_chart(fig, use_container_width=True)

        elif selection.chart_type == ChartType.LINE:
            from core.visualization.chart_builders.line_chart import build_line_chart
            fig = build_line_chart(selection)
            st.plotly_chart(fig, use_container_width=True)

        elif selection.chart_type == ChartType.PIE:
            from core.visualization.chart_builders.pie_chart import build_pie_chart
            fig = build_pie_chart(selection)
            st.plotly_chart(fig, use_container_width=True)

        elif selection.chart_type == ChartType.SCATTER:
            from core.visualization.chart_builders.scatter_chart import build_scatter_chart
            fig = build_scatter_chart(selection)
            st.plotly_chart(fig, use_container_width=True)

        elif selection.chart_type == ChartType.HISTOGRAM:
            from core.visualization.chart_builders.histogram_chart import build_histogram
            fig = build_histogram(selection)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        # Log but don't surface chart errors to the user. The data
        # table is already showing; a broken chart caption is noise.
        from logging_setup.logger import get_logger
        logger = get_logger(__name__)
        logger.warning("chart_rendering_failed", error=str(exc)[:200])
