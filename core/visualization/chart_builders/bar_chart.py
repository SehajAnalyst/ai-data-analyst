"""
core/visualization/chart_builders/bar_chart.py
================================================

Builds Plotly bar charts from a ChartSelection. Handles both
single-series (one y column) and grouped (multiple y columns).
Returns a Figure; never calls st.plotly_chart — rendering is
chart_renderer.py's responsibility.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.visualization.chart_selector import ChartSelection


def build_bar_chart(selection: ChartSelection) -> go.Figure:
    """
    Args:
        selection: ChartSelection with chart_type == BAR.
            selection.x_column: categorical axis column name.
            selection.y_columns: one or more numeric column names.

    Returns:
        Plotly Figure configured as a bar chart.
    """
    df = selection.shape.df
    x_col = selection.x_column
    y_cols = selection.y_columns

    fig = go.Figure()

    for y_col in y_cols:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y_col],
            name=y_col,
        ))

    fig.update_layout(
        title=selection.title,
        xaxis_title=x_col,
        yaxis_title=y_cols[0] if len(y_cols) == 1 else "Value",
        barmode="group",
        legend_title="Metric",
        margin=dict(t=60, b=40, l=40, r=20),
        height=400,
    )

    return fig
