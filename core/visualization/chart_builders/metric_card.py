"""
core/visualization/chart_builders/metric_card.py
==================================================

Returns structured data for SINGLE_METRIC results, not a Plotly
Figure. chart_renderer.py calls st.metric() with this, not
st.plotly_chart(). Kept in chart_builders/ because chart_selector.py
treats it as a selectable presentation type even though the rendering
path differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.visualization.chart_selector import ChartSelection


@dataclass
class MetricCardData:
    label: str
    value: str
    help_text: str


def build_metric_card(selection: ChartSelection) -> MetricCardData:
    """
    Extracts the single scalar value from the DataFrame and packages
    it for st.metric rendering.
    """
    df = selection.shape.df
    col = selection.shape.value_columns[0] if selection.shape.value_columns else df.columns[0]
    raw = df[col].iloc[0]

    # Format: integers without decimal, floats with two decimal places,
    # large numbers with thousands separators.
    if isinstance(raw, float):
        if raw == int(raw):
            formatted = f"{int(raw):,}"
        else:
            formatted = f"{raw:,.2f}"
    else:
        try:
            formatted = f"{int(raw):,}"
        except (TypeError, ValueError):
            formatted = str(raw)

    return MetricCardData(
        label=col.replace("_", " ").title(),
        value=formatted,
        help_text=f"Single aggregate value from column '{col}'.",
    )
