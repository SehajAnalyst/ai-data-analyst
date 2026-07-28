"""
core/visualization/chart_builders/pie_chart.py
================================================

Builds Plotly pie charts. Handles two cases:

1. CATEGORICAL_NUMERIC with one value column: slices are sized by the
   numeric column (e.g. revenue share by department).

2. SMALL_DISTRIBUTION with no numeric columns: slices are sized by
   the count of occurrences of each category value (e.g. how many
   employees per department).
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.visualization.chart_selector import ChartSelection


def build_pie_chart(selection: ChartSelection) -> go.Figure:
    """
    Args:
        selection: ChartSelection with chart_type == PIE.
            selection.x_column: label column.
            selection.y_columns: value column(s), or empty for count-based.

    Returns:
        Plotly Figure configured as a pie chart.
    """
    df = selection.shape.df
    label_col = selection.x_column

    if selection.y_columns:
        # Value-based: slice size = numeric column value.
        values = df[selection.y_columns[0]]
    else:
        # Count-based: slice size = frequency of each label.
        value_counts = df[label_col].value_counts()
        df = value_counts.reset_index()
        df.columns = [label_col, "count"]
        label_col = label_col
        values = df["count"]

    fig = go.Figure(go.Pie(
        labels=df[label_col],
        values=values,
        hole=0.3,           # donut style — easier to read than solid pie
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>",
    ))

    fig.update_layout(
        title=selection.title,
        margin=dict(t=60, b=20, l=20, r=20),
        height=400,
        showlegend=True,
    )

    return fig
