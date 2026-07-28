"""
core/visualization/chart_builders/scatter_chart.py
====================================================

Builds Plotly scatter plots for TWO_NUMERIC ResultShapes.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.visualization.chart_selector import ChartSelection


def build_scatter_chart(selection: ChartSelection) -> go.Figure:
    """
    Args:
        selection: ChartSelection with chart_type == SCATTER.
            selection.x_column: first numeric column (x-axis).
            selection.y_columns[0]: second numeric column (y-axis).

    Returns:
        Plotly Figure configured as a scatter plot.
    """
    df = selection.shape.df
    x_col = selection.x_column
    y_col = selection.y_columns[0]

    fig = go.Figure(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="markers",
        marker=dict(size=8, opacity=0.7),
        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
    ))

    fig.update_layout(
        title=selection.title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(t=60, b=40, l=40, r=20),
        height=400,
    )

    return fig
