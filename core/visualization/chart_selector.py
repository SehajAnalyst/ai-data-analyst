"""
core/visualization/chart_selector.py
========================================

Maps a ResultShape to a ChartSelection: which chart type, which
columns go on which axes, and what the title/description should say.

No Plotly dependency here. This module makes decisions; the builders
in chart_builders/ execute them. That split means the selection logic
can be unit-tested without constructing Plotly figures.

SELECTION RULES
---------------
EMPTY              -> no chart
SINGLE_METRIC      -> no chart (rendered as st.metric by result_table)
TIME_SERIES        -> line chart  (date x-axis, numeric y-axis)
CATEGORICAL_NUMERIC:
  categories <= 8  -> bar or pie (pie when single numeric value column)
  categories > 8   -> bar (pie with >8 slices is unreadable)
TWO_NUMERIC        -> scatter plot
SINGLE_NUMERIC     -> histogram
SMALL_DISTRIBUTION -> pie chart  (count of occurrences per category)
MULTI_DIMENSIONAL  -> no chart (table only)

WHY NO LLM FOR CHART SELECTION
--------------------------------
The shape→chart mapping is fully determined by column types and
cardinalities. An LLM adds latency, cost, and non-determinism to a
decision that three if-statements make correctly every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.execution.result_analyzer import ResultShape, ResultShapeType

_MAX_PIE_CATEGORIES = 8


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    NONE = "none"


@dataclass
class ChartSelection:
    """
    Everything a chart builder needs: which type, which columns, and
    display metadata. Produced by select_chart(); consumed by
    chart_renderer.py.
    """

    chart_type: ChartType
    x_column: str | None
    y_columns: list[str]           # may be >1 for grouped bar / multi-line
    title: str
    description: str
    shape: ResultShape             # carried for the builder to access df


def select_chart(shape: ResultShape) -> ChartSelection:
    """
    Returns a ChartSelection for a given ResultShape.

    Returns ChartType.NONE when no chart is appropriate. The renderer
    skips rendering in that case — callers should check
    selection.chart_type before building.
    """
    st = shape.shape_type

    if st == ResultShapeType.EMPTY:
        return _none(shape, "No data to visualise.")

    if st == ResultShapeType.SINGLE_METRIC:
        return _none(shape, "Single-value result — displayed as a metric card.")

    if st == ResultShapeType.MULTI_DIMENSIONAL:
        return _none(shape, "Too many dimensions — displaying as a table.")

    if st == ResultShapeType.TIME_SERIES:
        return _time_series(shape)

    if st == ResultShapeType.CATEGORICAL_NUMERIC:
        return _categorical_numeric(shape)

    if st == ResultShapeType.TWO_NUMERIC:
        return _two_numeric(shape)

    if st == ResultShapeType.SINGLE_NUMERIC:
        return _single_numeric(shape)

    if st == ResultShapeType.SMALL_DISTRIBUTION:
        return _small_distribution(shape)

    return _none(shape, "Could not determine an appropriate chart type.")


# ── Selection helpers ─────────────────────────────────────────────────────


def _time_series(shape: ResultShape) -> ChartSelection:
    time_col = shape.time_column

    val_cols = [
        col
        for col in shape.value_columns
        if col not in {"sale_id", "product_id"}
    ]

    y_label = ", ".join(val_cols)
    return ChartSelection(
        chart_type=ChartType.LINE,
        x_column=time_col,
        y_columns=val_cols,
        title=f"{y_label} over time",
        description=f"Line chart showing {y_label} by {time_col}.",
        shape=shape,
    )


def _categorical_numeric(shape: ResultShape) -> ChartSelection:
    cat_col = shape.category_column

    # Exclude identifier columns from chart metrics.
    val_cols = [
        col
        for col in shape.value_columns
        if col not in {"sale_id", "product_id"}
    ]
    n_categories = shape.df[cat_col].nunique() if cat_col else 0

    # Pie only when single numeric column AND few enough categories.
    if len(val_cols) == 1 and n_categories <= _MAX_PIE_CATEGORIES:
        val_col = val_cols[0]
        return ChartSelection(
            chart_type=ChartType.PIE,
            x_column=cat_col,
            y_columns=val_cols,
            title=f"{val_col} by {cat_col}",
            description=f"Pie chart showing the share of {val_col} per {cat_col}.",
            shape=shape,
        )

    y_label = ", ".join(val_cols)
    return ChartSelection(
        chart_type=ChartType.BAR,
        x_column=cat_col,
        y_columns=val_cols,
        title=f"{y_label} by {cat_col}",
        description=f"Bar chart comparing {y_label} across {cat_col} categories.",
        shape=shape,
    )


def _two_numeric(shape: ResultShape) -> ChartSelection:
    x_col, y_col = shape.value_columns[0], shape.value_columns[1]
    return ChartSelection(
        chart_type=ChartType.SCATTER,
        x_column=x_col,
        y_columns=[y_col],
        title=f"{y_col} vs {x_col}",
        description=f"Scatter plot showing the relationship between {x_col} and {y_col}.",
        shape=shape,
    )


def _single_numeric(shape: ResultShape) -> ChartSelection:
    col = shape.value_columns[0]
    return ChartSelection(
        chart_type=ChartType.HISTOGRAM,
        x_column=col,
        y_columns=[],
        title=f"Distribution of {col}",
        description=f"Histogram showing the distribution of {col} values.",
        shape=shape,
    )


def _small_distribution(shape: ResultShape) -> ChartSelection:
    cat_col = shape.category_column
    return ChartSelection(
        chart_type=ChartType.PIE,
        x_column=cat_col,
        y_columns=[],
        title=f"Distribution of {cat_col}",
        description=f"Pie chart showing count of occurrences per {cat_col}.",
        shape=shape,
    )


def _none(shape: ResultShape, reason: str) -> ChartSelection:
    return ChartSelection(
        chart_type=ChartType.NONE,
        x_column=None,
        y_columns=[],
        title="",
        description=reason,
        shape=shape,
    )
