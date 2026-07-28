"""
core/insights/insight_generator.py
=====================================

Generates plain-English business insight narrative from query
results: a summary, key trends, outliers, important metrics, and
three follow-up questions.

DESIGN PRINCIPLE: THE LLM NARRATES, IT DOES NOT COMPUTE
--------------------------------------------------------------
This mirrors the same boundary used everywhere else in this codebase
(sql_validator.py validates deterministically; the LLM only proposes
SQL). Here: outlier detection, min/max/mean/median, and value counts
are all computed in plain Python via pandas BEFORE the LLM is called.
The LLM's job is to phrase these pre-computed facts in plain English —
it is explicitly instructed not to invent or recalculate numbers.

Why this matters concretely: asking an LLM "look at this data and
tell me the outliers" from a sample of rows or from vague stats
invites fabrication — the model will describe something as
unusual that isn't, or miss a real outlier that IS unusual, because
it's pattern-matching on prose, not doing arithmetic. Computing
outliers with a real IQR (interquartile range) check first and
handing the LLM the flagged values as fact removes that whole failure
mode. The LLM cannot claim an outlier exists that wasn't in the
DATA SUMMARY, because the prompt explicitly forbids it and the
summary is the only source of numbers it has.

WHY THIS IS A SEPARATE LLM CALL FROM SQL GENERATION
-----------------------------------------------------------
The insight narrative needs the actual query RESULTS, which don't
exist until after execution — SQL generation happens before execution.
They are sequentially dependent, not just conceptually separable.
This also means insight generation can run after results are already
shown to the user (see app/pipeline.py) without blocking the
result-display path.

EMPTY RESULTS
--------------
Zero-row results skip the LLM call entirely. There's nothing to
narrate, and burning an API call to have a model describe an empty
table adds latency and cost for no value. A fixed response is
returned instead, with generic (not data-derived) follow-up
suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.execution.query_executor import QueryResult
from core.llm.llm_client_factory import get_llm_provider
from core.llm.llm_config_loader import resolve_insight_generation_config
from core.nl2sql.prompt_builder import build_general_prompt
from exceptions.domain_exceptions import LLMAPIError, LLMTimeoutError
from logging_setup.logger import LogCategory, get_logger

logger = get_logger(__name__)

_MAX_CATEGORICAL_VALUES_SHOWN = 5
_MIN_ROWS_FOR_OUTLIER_CHECK = 4   # IQR is not meaningful on tiny samples
_MAX_OUTLIER_EXAMPLES_PER_COLUMN = 3


@dataclass
class InsightResult:
    """
    Complete output of one generate_insight() call.

    is_empty distinguishes "zero rows, no LLM call was made" from a
    normal result — callers (pipeline.py) can render a lighter-weight
    UI treatment for the empty case rather than a full insights panel.
    """

    summary: str
    key_trends: list[str] = field(default_factory=list)
    outliers: list[str] = field(default_factory=list)
    important_metrics: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    is_empty: bool = False


def generate_insight(
    user_question: str,
    generated_sql: str,
    result: QueryResult,
) -> InsightResult:
    """
    Produces a plain-English business insight from a query result.

    Args:
        user_question: original natural-language question.
        generated_sql: the SQL that was executed (gives the LLM
            grounding on what was actually measured).
        result: query result to analyze.

    Returns:
        InsightResult. Never raises for LLM failures — on any LLM
        error, returns a minimal InsightResult with a generic summary
        and empty lists, so a failed insight call degrades gracefully
        rather than breaking the caller's pipeline. This matches the
        "optional step" framing: insight generation failing must not
        prevent the user from seeing their query results.

    Raises:
        Nothing. All exceptions are caught internally and logged.
    """
    log = logger.bind(category=LogCategory.LLM_CALL, insight_call=True)

    if result.row_count == 0:
        log.info("insight_generation_skipped_empty_result")
        return _empty_result_insight(user_question)

    df = pd.DataFrame(result.rows, columns=result.columns)
    data_summary = _build_data_summary(df, result)

    try:
        provider = get_llm_provider()
        config = resolve_insight_generation_config()

        system_prompt, user_message = build_general_prompt(
            template_filename="business_insights.txt",
            variables={
                "user_question": user_question,
                "generated_sql": generated_sql,
                "data_summary": data_summary,
            },
            user_message="Generate the business insights as JSON.",
        )

        response = provider.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        parsed = _parse_insight_response(response.content)
        if parsed is None:
            log.warning("insight_response_parse_failed")
            return _fallback_insight()

        log.info("insight_generation_succeeded")
        return parsed

    except (LLMAPIError, LLMTimeoutError) as exc:
        log.warning("insight_generation_llm_failed", error=str(exc)[:200])
        return _fallback_insight()
    except Exception as exc:
        log.warning("insight_generation_unexpected_error", error=str(exc)[:200])
        return _fallback_insight()


# ---------------------------------------------------------------------------
# Deterministic data summarization (no LLM involved)
# ---------------------------------------------------------------------------


def _build_data_summary(df: pd.DataFrame, result: QueryResult) -> str:
    """
    Computes a compact, fact-only text summary of the result set:
    row/column counts, numeric column stats, categorical value counts,
    and IQR-based outlier detection.

    This is the ONLY source of numbers the LLM prompt receives — see
    module docstring on why this boundary exists. Every number in the
    final InsightResult must trace back to something computed here.
    """
    lines: list[str] = [
        f"Total rows: {result.row_count}" + (" (truncated — more rows existed)" if result.truncated else ""),
        f"Columns: {', '.join(df.columns)}",
        "",
    ]

    numeric_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "f", "u")]
    categorical_cols = [c for c in df.columns if df[c].dtype.kind == "O"]

    if numeric_cols:
        lines.append("Numeric column statistics:")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            lines.append(
                f"  - {col}: min={series.min():.2f}, max={series.max():.2f}, "
                f"mean={series.mean():.2f}, median={series.median():.2f}"
            )
        lines.append("")

    outlier_lines = _detect_outliers(df, numeric_cols)
    if outlier_lines:
        lines.append("Detected outliers (IQR method — values far outside the typical range):")
        lines.extend(f"  - {line}" for line in outlier_lines)
        lines.append("")
    else:
        lines.append("Detected outliers: none.")
        lines.append("")

    if categorical_cols:
        lines.append("Categorical column value counts (top values):")
        for col in categorical_cols:
            cardinality = df[col].nunique()
            if cardinality > 30:
                lines.append(f"  - {col}: {cardinality} distinct values (too many to list)")
                continue
            counts = df[col].value_counts().head(_MAX_CATEGORICAL_VALUES_SHOWN)
            formatted = ", ".join(f"{val} ({count})" for val, count in counts.items())
            lines.append(f"  - {col}: {formatted}")

    return "\n".join(lines)


def _detect_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    """
    Flags outliers per numeric column using the standard IQR method:
    a value is an outlier if it falls outside
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Returns human-readable strings describing each column's outliers,
    capped at _MAX_OUTLIER_EXAMPLES_PER_COLUMN examples per column to
    keep the prompt compact. Skips columns with fewer than
    _MIN_ROWS_FOR_OUTLIER_CHECK non-null values — IQR on 2-3 points is
    not statistically meaningful and would produce noise, not signal.
    """
    results: list[str] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < _MIN_ROWS_FOR_OUTLIER_CHECK:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue  # no spread — every value identical, no outliers possible

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        if outliers.empty:
            continue

        examples = outliers.sort_values(ascending=False).head(_MAX_OUTLIER_EXAMPLES_PER_COLUMN)
        example_str = ", ".join(f"{v:.2f}" for v in examples)
        results.append(
            f"{col}: {len(outliers)} outlier value(s) found, "
            f"outside the normal range of {lower_bound:.2f}–{upper_bound:.2f}. "
            f"Example value(s): {example_str}"
        )

    return results


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_insight_response(raw_content: str) -> InsightResult | None:
    """
    Parses the LLM's JSON response into an InsightResult.

    Uses the same tolerant-parsing approach as
    core/nl2sql/sql_generator.py (strip markdown code fences, fall
    back to extracting an embedded JSON object from surrounding
    prose) — duplicated here rather than imported, since that parsing
    logic in sql_generator.py is a private, SQL-specific helper not
    intended as a shared utility. If a third consumer needs this same
    parsing later, it should be extracted into a shared module then.

    Returns None if parsing fails entirely.
    """
    import json
    import re

    cleaned = raw_content.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    try:
        return InsightResult(
            summary=str(data.get("summary", "")).strip(),
            key_trends=_as_str_list(data.get("key_trends")),
            outliers=_as_str_list(data.get("outliers")),
            important_metrics=_as_str_list(data.get("important_metrics")),
            follow_up_questions=_as_str_list(data.get("follow_up_questions"))[:3],
        )
    except (TypeError, ValueError):
        return None


def _as_str_list(value: object) -> list[str]:
    """Coerces a JSON value into a list of strings, tolerating a model
    that returns a single string instead of a list for a field that
    should be a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


# ---------------------------------------------------------------------------
# Fallback / empty-result insights (no LLM call)
# ---------------------------------------------------------------------------


def _empty_result_insight(user_question: str) -> InsightResult:
    """Fixed response for zero-row results — no LLM call made. See
    module docstring for why an empty result skips generation
    entirely rather than asking the LLM to describe nothing."""
    return InsightResult(
        summary="No data matched this query.",
        key_trends=[],
        outliers=[],
        important_metrics=[],
        follow_up_questions=[
            "Try broadening the filters in your question.",
            "Check whether the date range or category you asked about actually has data.",
            "Ask a more general question to see what data is available.",
        ],
        is_empty=True,
    )


def _fallback_insight() -> InsightResult:
    """Returned when the LLM call fails or its response can't be
    parsed. Keeps the pipeline non-fatal — the user still sees their
    query results even if the insight narrative couldn't be generated."""
    return InsightResult(
        summary="Insight generation is temporarily unavailable for this result.",
        key_trends=[],
        outliers=[],
        important_metrics=[],
        follow_up_questions=[],
    )
