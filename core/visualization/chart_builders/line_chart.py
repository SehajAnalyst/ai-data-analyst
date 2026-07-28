"""
core/visualization/chart_builders/line_chart.py
=================================================

Builds Plotly line charts for time-series ResultShapes.
One trace per y column; markers enabled so individual data points
are visible on sparse series.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.visualization.chart_selector import ChartSelection


def build_line_chart(selection: ChartSelection) -> go.Figure:
    """
    Args:
        selection: ChartSelection with chart_type == LINE.
            selection.x_column: date/time column (x-axis).
            selection.y_columns: one or more numeric column names.

    Returns:
        Plotly Figure configured as a line chart.
    """
    df = selection.shape.df
    x_col = selection.x_column
    y_cols = selection.y_columns

    fig = go.Figure()

    for y_col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines+markers",
            name=y_col,
        ))

    fig.update_layout(
        title=selection.title,
        xaxis_title=x_col,
        yaxis_title=y_cols[0] if len(y_cols) == 1 else "Value",
        legend_title="Metric",
        margin=dict(t=60, b=40, l=40, r=20),
        height=400,
    )

    return fig
