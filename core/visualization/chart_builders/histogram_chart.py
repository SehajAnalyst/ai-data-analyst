"""
core/visualization/chart_builders/histogram_chart.py
======================================================

Builds Plotly histograms for SINGLE_NUMERIC ResultShapes.
nbinsx is left to Plotly's auto-binning (Sturges' rule) rather than
hardcoded — it adjusts to the actual data distribution.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.visualization.chart_selector import ChartSelection


def build_histogram(selection: ChartSelection) -> go.Figure:
    """
    Args:
        selection: ChartSelection with chart_type == HISTOGRAM.
            selection.x_column: the numeric column to distribute.

    Returns:
        Plotly Figure configured as a histogram.
    """
    df = selection.shape.df
    col = selection.x_column

    fig = go.Figure(go.Histogram(
        x=df[col],
        name=col,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title=selection.title,
        xaxis_title=col,
        yaxis_title="Count",
        bargap=0.05,
        margin=dict(t=60, b=40, l=40, r=20),
        height=400,
    )

    return fig
