"""
tests/unit/core/test_visualization.py
=======================================

Tests the full visualization chain:
  QueryResult → analyze_result → select_chart → build_*

No Streamlit calls here. chart_renderer.py is the only Streamlit-aware
file in the visualization stack and is tested via integration.
"""

from __future__ import annotations

import pandas as pd

from core.execution.query_executor import QueryResult
from core.execution.result_analyzer import ResultShapeType, analyze_result
from core.visualization.chart_selector import ChartType, select_chart


def make_result(data: dict) -> QueryResult:
    df = pd.DataFrame(data)
    return QueryResult(
        columns=list(df.columns),
        rows=[tuple(r) for r in df.values],
        row_count=len(df),
        truncated=False,
        execution_time_ms=1.0,
    )


# ── Shape detection ────────────────────────────────────────────────────────


class TestShapeDetection:
    def test_empty_result(self):
        result = QueryResult(columns=["x"], rows=[], row_count=0,
                             truncated=False, execution_time_ms=0)
        assert analyze_result(result).shape_type == ResultShapeType.EMPTY

    def test_single_metric(self):
        shape = analyze_result(make_result({"count": [42]}))
        assert shape.shape_type == ResultShapeType.SINGLE_METRIC

    def test_categorical_numeric(self):
        shape = analyze_result(make_result({
            "department": ["Eng", "Mkt", "HR"],
            "avg_salary": [90000.0, 70000.0, 61000.0],
        }))
        assert shape.shape_type == ResultShapeType.CATEGORICAL_NUMERIC
        assert shape.category_column == "department"
        assert "avg_salary" in shape.value_columns

    def test_time_series_with_date_column(self):
        shape = analyze_result(make_result({
            "sale_date": ["2024-01", "2024-02", "2024-03"],
            "revenue": [10000.0, 12000.0, 9500.0],
        }))
        assert shape.shape_type == ResultShapeType.TIME_SERIES
        assert shape.time_column == "sale_date"

    def test_time_series_with_hire_date(self):
        shape = analyze_result(make_result({
            "hire_date": ["2021-03-01", "2022-01-15", "2023-06-01"],
            "salary": [95000.0, 85000.0, 72000.0],
        }))
        assert shape.shape_type == ResultShapeType.TIME_SERIES

    def test_two_numeric_columns(self):
        shape = analyze_result(make_result({
            "experience": [1, 3, 5, 7, 10],
            "salary": [50000, 65000, 80000, 95000, 120000],
        }))
        assert shape.shape_type == ResultShapeType.TWO_NUMERIC
        assert len(shape.value_columns) == 2

    def test_single_numeric_column(self):
        shape = analyze_result(make_result({
            "salary": [50000, 65000, 80000, 95000, 110000],
        }))
        assert shape.shape_type == ResultShapeType.SINGLE_NUMERIC

    def test_small_distribution(self):
        shape = analyze_result(make_result({
            "department": ["Eng", "Mkt", "HR", "Finance"],
        }))
        assert shape.shape_type == ResultShapeType.SMALL_DISTRIBUTION

    def test_non_date_object_column_not_flagged_as_date(self):
        shape = analyze_result(make_result({
            "employee_name": ["Alice", "Bob", "Carol"],
            "salary": [90000.0, 80000.0, 70000.0],
        }))
        # employee_name has no date hint → should be categorical, not time_series
        assert shape.shape_type == ResultShapeType.CATEGORICAL_NUMERIC
        assert shape.time_column is None

    def test_high_cardinality_text_not_categorical(self):
        # 60 unique values → exceeds _MAX_CATEGORICAL_CARDINALITY
        shape = analyze_result(make_result({
            "name": [f"Person_{i}" for i in range(60)],
            "salary": [float(50000 + i * 1000) for i in range(60)],
        }))
        # With no usable categorical column, falls to TWO_NUMERIC or SINGLE_NUMERIC
        assert shape.shape_type in (
            ResultShapeType.SINGLE_NUMERIC,
            ResultShapeType.TWO_NUMERIC,
            ResultShapeType.MULTI_DIMENSIONAL,
        )


# ── Chart selection ────────────────────────────────────────────────────────


class TestChartSelection:
    def test_categorical_numeric_bar(self):
        result = make_result({
            "department": ["Eng", "Mkt", "HR", "Finance", "Legal", "Ops", "Sales", "IT", "PM"],
            "avg_salary": [90000.0] * 9,
        })
        shape = analyze_result(result)
        selection = select_chart(shape)
        # 9 categories → bar, not pie
        assert selection.chart_type == ChartType.BAR

    def test_categorical_numeric_pie_when_few_categories(self):
        result = make_result({
            "department": ["Eng", "Mkt", "HR"],
            "avg_salary": [90000.0, 70000.0, 61000.0],
        })
        shape = analyze_result(result)
        selection = select_chart(shape)
        assert selection.chart_type == ChartType.PIE

    def test_time_series_line(self):
        result = make_result({
            "sale_date": ["2024-01", "2024-02", "2024-03"],
            "revenue": [10000.0, 12000.0, 9500.0],
        })
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.LINE

    def test_two_numeric_scatter(self):
        result = make_result({
            "experience": [1, 3, 5, 7, 10],
            "salary": [50000, 65000, 80000, 95000, 120000],
        })
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.SCATTER

    def test_single_numeric_histogram(self):
        result = make_result({
            "salary": [50000, 65000, 80000, 95000, 110000],
        })
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.HISTOGRAM

    def test_small_distribution_pie(self):
        result = make_result({
            "department": ["Eng", "Mkt", "HR", "Finance"],
        })
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.PIE

    def test_empty_no_chart(self):
        result = QueryResult(columns=["x"], rows=[], row_count=0,
                             truncated=False, execution_time_ms=0)
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.NONE

    def test_selection_has_title_and_description(self):
        result = make_result({
            "department": ["Eng", "Mkt"],
            "avg_salary": [90000.0, 70000.0],
        })
        selection = select_chart(analyze_result(result))
        assert selection.title
        assert selection.description


# ── Figure construction ────────────────────────────────────────────────────


class TestFigureBuilders:
    """Verify builders produce Plotly Figure objects without errors."""

    def test_bar_chart_builds(self):
        import plotly.graph_objects as go
        from core.visualization.chart_builders.bar_chart import build_bar_chart
        result = make_result({"dept": ["A", "B"], "val": [100.0, 200.0]})
        selection = select_chart(analyze_result(result))
        fig = build_bar_chart(selection)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_line_chart_builds(self):
        import plotly.graph_objects as go
        from core.visualization.chart_builders.line_chart import build_line_chart
        result = make_result({
            "sale_date": ["2024-01", "2024-02", "2024-03"],
            "revenue": [1000.0, 1200.0, 950.0],
        })
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.LINE
        fig = build_line_chart(selection)
        assert isinstance(fig, go.Figure)

    def test_pie_chart_builds_value_based(self):
        import plotly.graph_objects as go
        from core.visualization.chart_builders.pie_chart import build_pie_chart
        result = make_result({"dept": ["A", "B", "C"], "budget": [100.0, 200.0, 150.0]})
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.PIE
        fig = build_pie_chart(selection)
        assert isinstance(fig, go.Figure)

    def test_pie_chart_builds_count_based(self):
        import plotly.graph_objects as go
        from core.visualization.chart_builders.pie_chart import build_pie_chart
        result = make_result({"department": ["Eng", "Mkt", "HR"]})
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.PIE
        fig = build_pie_chart(selection)
        assert isinstance(fig, go.Figure)

    def test_scatter_chart_builds(self):
        import plotly.graph_objects as go
        from core.visualization.chart_builders.scatter_chart import build_scatter_chart
        result = make_result({"exp": [1, 2, 3, 4], "sal": [40000, 50000, 60000, 70000]})
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.SCATTER
        fig = build_scatter_chart(selection)
        assert isinstance(fig, go.Figure)

    def test_histogram_builds(self):
        import plotly.graph_objects as go
        from core.visualization.chart_builders.histogram_chart import build_histogram
        result = make_result({"salary": [50000, 60000, 70000, 80000, 90000]})
        selection = select_chart(analyze_result(result))
        assert selection.chart_type == ChartType.HISTOGRAM
        fig = build_histogram(selection)
        assert isinstance(fig, go.Figure)

    def test_metric_card_builds(self):
        from core.visualization.chart_builders.metric_card import (
            MetricCardData,
            build_metric_card,
        )
        result = make_result({"total_employees": [42]})
        # SINGLE_METRIC maps to NONE — metric card is rendered by result_table.py
        # Test the builder directly
        shape = analyze_result(result)
        from core.visualization.chart_selector import ChartSelection, ChartType
        manual_selection = ChartSelection(
            chart_type=ChartType.NONE,
            x_column=None,
            y_columns=["total_employees"],
            title="",
            description="",
            shape=shape,
        )
        card = build_metric_card(manual_selection)
        assert isinstance(card, MetricCardData)
        assert card.value == "42"
