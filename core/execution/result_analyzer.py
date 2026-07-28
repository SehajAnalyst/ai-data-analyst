"""
core/execution/result_analyzer.py
====================================

Classifies a QueryResult's shape into one of five chartable types.
All logic is deterministic Python — no LLM, no Plotly dependency.

COLUMN TYPE DETECTION RULES (verified against real pandas behavior)
---------------------------------------------------------------------

Numeric: dtype.kind in ('i', 'f', 'u') — integers, floats, unsigned.

Date: TWO conditions must both be true to avoid false positives:
  1. Column name contains a date-hint word (date, month, year, time,
     period, quarter, week, day). This guards against a numeric
     "year" column being misread as a time series.
  2. >80% of non-null values parse successfully via pd.to_datetime.
  SQLite stores dates as TEXT, so raw dtype is 'object'. We can't
  rely on dtype alone — we must attempt parse.

Categorical: object dtype that is NOT a detected date column, AND
  has cardinality <= 50 (high-cardinality text is not useful to
  group by visually).

SHAPE CLASSIFICATION ORDER
---------------------------
  1. EMPTY          — zero rows
  2. SINGLE_METRIC  — 1 row, 1 numeric column (e.g. COUNT(*))
  3. TIME_SERIES    — many rows, >=1 date col + >=1 numeric col
  4. CATEGORICAL_NUMERIC — many rows, 1 categorical + 1-3 numerics
  5. TWO_NUMERIC    — 2 numeric columns (scatter plot candidate)
  6. SINGLE_NUMERIC — 1 numeric column (histogram candidate)
  7. SMALL_DISTRIBUTION — 1 categorical column only, <=10 unique vals
  8. MULTI_DIMENSIONAL — everything else (table only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from core.execution.query_executor import QueryResult

# Words in a column name that suggest it holds date/time data.
_DATE_HINT_PATTERN = re.compile(
    r'(?:^|_)(date|month|year|time|period|quarter|week|day|timestamp|created|updated)(?:_|$)',
    re.IGNORECASE,
)

# Threshold: what fraction of non-null values must parse as dates.
_DATE_PARSE_THRESHOLD = 0.80

# Categorical column cardinality cap. Above this the column is too
# high-cardinality to be useful as a chart axis label.
_MAX_CATEGORICAL_CARDINALITY = 50

# Pie chart specific: only use pie when category count is this small.
_MAX_PIE_CATEGORIES = 8


class ResultShapeType(str, Enum):
    EMPTY = "empty"
    SINGLE_METRIC = "single_metric"
    TIME_SERIES = "time_series"
    CATEGORICAL_NUMERIC = "categorical_numeric"
    TWO_NUMERIC = "two_numeric"
    SINGLE_NUMERIC = "single_numeric"
    SMALL_DISTRIBUTION = "small_distribution"
    MULTI_DIMENSIONAL = "multi_dimensional"


@dataclass
class ResultShape:
    """
    Structural classification of one query result, used by
    chart_selector.py to pick a chart type without knowing anything
    about Plotly.
    """

    shape_type: ResultShapeType
    category_column: str | None       # best categorical column for x-axis
    value_columns: list[str]          # numeric columns for y-axis/values
    time_column: str | None           # detected date column if any
    df: pd.DataFrame = field(repr=False)   # the actual data, carried forward
                                            # to avoid rebuilding it in builders


def analyze_result(result: QueryResult) -> ResultShape:
    """
    Converts a QueryResult into a ResultShape by inspecting the
    DataFrame's column types and cardinalities.

    Args:
        result: a QueryResult from query_executor.execute_query().

    Returns:
        ResultShape. chart_selector.py maps this to a ChartType.
    """
    if result.row_count == 0:
        df = pd.DataFrame(columns=result.columns)
        return ResultShape(
            shape_type=ResultShapeType.EMPTY,
            category_column=None,
            value_columns=[],
            time_column=None,
            df=df,
        )

    df = pd.DataFrame(result.rows, columns=result.columns)

    date_cols = _detect_date_columns(df)
    numeric_cols = _detect_numeric_columns(df)
    categorical_cols = _detect_categorical_columns(df, date_cols)

    # ── 1-row single numeric: a scalar aggregate ───────────────────────────
    if result.row_count == 1 and len(numeric_cols) == 1 and len(df.columns) == 1:
        return ResultShape(
            shape_type=ResultShapeType.SINGLE_METRIC,
            category_column=None,
            value_columns=numeric_cols,
            time_column=None,
            df=df,
        )

    # ── Time series: date column present with at least one numeric ─────────
    if date_cols and numeric_cols:
        return ResultShape(
            shape_type=ResultShapeType.TIME_SERIES,
            category_column=None,
            value_columns=numeric_cols,
            time_column=date_cols[0],
            df=df,
        )

    # ── Categorical + numeric: bar chart territory ─────────────────────────
    if categorical_cols and numeric_cols:
        return ResultShape(
            shape_type=ResultShapeType.CATEGORICAL_NUMERIC,
            category_column=categorical_cols[0],
            value_columns=numeric_cols[:3],  # cap at 3 series for readability
            time_column=None,
            df=df,
        )

    # ── Two numeric columns: scatter plot ──────────────────────────────────
    if len(numeric_cols) == 2:
        return ResultShape(
            shape_type=ResultShapeType.TWO_NUMERIC,
            category_column=None,
            value_columns=numeric_cols,
            time_column=None,
            df=df,
        )

    # ── Single numeric column: histogram ──────────────────────────────────
    if len(numeric_cols) == 1:
        return ResultShape(
            shape_type=ResultShapeType.SINGLE_NUMERIC,
            category_column=None,
            value_columns=numeric_cols,
            time_column=None,
            df=df,
        )

    # ── Single categorical, small cardinality: pie chart ──────────────────
    if len(categorical_cols) == 1 and len(numeric_cols) == 0:
        cardinality = df[categorical_cols[0]].nunique()
        if cardinality <= _MAX_PIE_CATEGORIES:
            return ResultShape(
                shape_type=ResultShapeType.SMALL_DISTRIBUTION,
                category_column=categorical_cols[0],
                value_columns=[],
                time_column=None,
                df=df,
            )

    # ── Everything else: show a table, no chart ────────────────────────────
    return ResultShape(
        shape_type=ResultShapeType.MULTI_DIMENSIONAL,
        category_column=None,
        value_columns=numeric_cols,
        time_column=None,
        df=df,
    )


# ── Column type detection helpers ─────────────────────────────────────────


def _detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns column names whose dtype is numeric (int, float, unsigned).
    Uses pandas dtype.kind rather than str(dtype) to be robust across
    numpy integer subtypes (int32, int64, etc.).
    """
    return [
        col for col in df.columns
        if df[col].dtype.kind in ('i', 'f', 'u')
    ]


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns column names that look like date/time data.

    Requires BOTH conditions:
      1. Column name contains a date-hint word.
      2. >80% of non-null values parse as dates via pd.to_datetime.

    Condition 1 guards against integer columns named 'year' being
    mistaken for time series (every 4-digit integer parses as a date).
    Condition 2 guards against non-date text columns with date-hint
    names.

    Columns already typed as datetime (dtype.kind == 'M') are
    included regardless of name — they were typed correctly at source.
    """
    date_cols = []
    for col in df.columns:
        series = df[col]

        # Already a datetime dtype — include unconditionally.
        if series.dtype.kind == 'M':
            date_cols.append(col)
            continue

        # Object dtype: check name hint + parse rate.
        if series.dtype.kind == 'O' and _DATE_HINT_PATTERN.search(col):
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            parsed = pd.to_datetime(non_null, errors='coerce')
            parse_rate = parsed.notna().sum() / len(non_null)
            if parse_rate >= _DATE_PARSE_THRESHOLD:
                # Mutate the DataFrame in place for downstream builders.
                df[col] = pd.to_datetime(series, errors='coerce')
                date_cols.append(col)

    return date_cols


def _detect_categorical_columns(
    df: pd.DataFrame,
    date_cols: list[str],
) -> list[str]:
    """
    Returns object-dtype columns that are not date columns and have
    cardinality <= _MAX_CATEGORICAL_CARDINALITY.

    High-cardinality text columns (names, IDs, free-text) are excluded
    because they produce unreadable chart axes and aren't useful for
    grouping. A column with 300 unique product names is a table column,
    not a bar chart axis.
    """
    date_col_set = set(date_cols)
    return [
        col for col in df.columns
        if (
            col not in date_col_set
            and df[col].dtype.kind == 'O'
            and df[col].nunique() <= _MAX_CATEGORICAL_CARDINALITY
        )
    ]
