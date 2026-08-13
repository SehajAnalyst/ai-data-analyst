"""
Renders Plotly charts into the Streamlit UI.

The only file in the project that both imports Plotly builders AND
calls Streamlit rendering functions. All other visualization code
is Streamlit-free; this is where those two worlds meet.

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

    Silently skips rendering when no chart is appropriate.
    Never raises — a visualization failure must not crash the chat page.
    """

    if result.row_count == 0:
        return

    try:
        shape = analyze_result(result)
        selection = select_chart(shape)

        if selection.chart_type == ChartType.NONE:
            return

        st.caption(f"📊 {selection.description}")

        # Unique key prevents duplicate Streamlit element IDs
        # when multiple charts are rendered.
        chart_key = f"chart_{id(result)}_{id(selection)}"

        if selection.chart_type == ChartType.BAR:
            from core.visualization.chart_builders.bar_chart import build_bar_chart

            fig = build_bar_chart(selection)

            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{chart_key}_bar",
            )

        elif selection.chart_type == ChartType.LINE:
            from core.visualization.chart_builders.line_chart import build_line_chart

            fig = build_line_chart(selection)

            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{chart_key}_line",
            )

        elif selection.chart_type == ChartType.PIE:
            from core.visualization.chart_builders.pie_chart import build_pie_chart

            fig = build_pie_chart(selection)

            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{chart_key}_pie",
            )

        elif selection.chart_type == ChartType.SCATTER:
            from core.visualization.chart_builders.scatter_chart import build_scatter_chart

            fig = build_scatter_chart(selection)

            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{chart_key}_scatter",
            )

        elif selection.chart_type == ChartType.HISTOGRAM:
            from core.visualization.chart_builders.histogram_chart import build_histogram

            fig = build_histogram(selection)

            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{chart_key}_histogram",
            )

    except Exception as exc:
        st.error(f"Chart rendering failed: {exc}")
        st.exception(exc)