from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

import pandas as pd

from core.execution.query_executor import QueryResult
from core.llm.llm_client_factory import get_llm_provider
from core.llm.llm_config_loader import resolve_insight_generation_config
from core.nl2sql.prompt_builder import build_general_prompt
from exceptions.domain_exceptions import LLMAPIError, LLMTimeoutError
from logging_setup.logger import LogCategory, get_logger


logger = get_logger(__name__)

_MAX_CATEGORICAL_VALUES_SHOWN = 5
_MIN_ROWS_FOR_OUTLIER_CHECK = 4
_MAX_OUTLIER_EXAMPLES_PER_COLUMN = 3


@dataclass
class InsightResult:
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

    log = logger.bind(
        category=LogCategory.LLM_CALL,
        insight_call=True,
    )

    # ---------------------------------------------------------
    # EMPTY RESULT
    # ---------------------------------------------------------
    if result.row_count == 0:
        log.info("insight_generation_skipped_empty_result")

        return _empty_result_insight(user_question)

    # ---------------------------------------------------------
    # BUILD DETERMINISTIC DATA SUMMARY
    # ---------------------------------------------------------
    try:
        df = pd.DataFrame(
            result.rows,
            columns=result.columns,
        )

        data_summary = _build_data_summary(df, result)
        print("\n===== DATA SUMMARY SENT TO LLM =====")
        print(data_summary)
        print("====================================\n")

    except Exception as exc:
        log.warning(
            "insight_data_summary_failed",
            error=str(exc)[:200],
        )

        return _fallback_insight(
            result=result,
            user_question=user_question,
        )

    # ---------------------------------------------------------
    # CALL LLM
    # ---------------------------------------------------------
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
            user_message=(
                "Generate the business insights as JSON. "
                "Use ONLY the facts provided in DATA SUMMARY."
            ),
        )

        response = provider.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        print("\n===== RAW INSIGHT LLM RESPONSE =====")
        print(response.content)
        print("====================================\n")
        parsed = _parse_insight_response(response.content)

        if parsed is None:
            log.warning("insight_response_parse_failed")

            return _fallback_insight(
                result=result,
                user_question=user_question,
            )

        # -----------------------------------------------------
        # MAKE SURE SUMMARY EXISTS
        # -----------------------------------------------------
        if not parsed.summary.strip():
            parsed.summary = _build_basic_summary(
                result,
                user_question,
            )

        # -----------------------------------------------------
        # MAKE SURE FOLLOW-UP QUESTIONS EXIST
        # -----------------------------------------------------
        if not parsed.follow_up_questions:
            parsed.follow_up_questions = _default_follow_up_questions(
                result
            )

        # Maximum 3 questions
        parsed.follow_up_questions = parsed.follow_up_questions[:3]

        parsed.is_empty = False

        log.info(
            "insight_generation_succeeded",
            summary_present=bool(parsed.summary),
            trends_count=len(parsed.key_trends),
            outliers_count=len(parsed.outliers),
            metrics_count=len(parsed.important_metrics),
            followups_count=len(parsed.follow_up_questions),
        )

        return parsed

    except (LLMAPIError, LLMTimeoutError) as exc:

        log.warning(
            "insight_generation_llm_failed",
            error=str(exc)[:200],
        )

        return _fallback_insight(
            result=result,
            user_question=user_question,
        )

    except Exception as exc:

        log.warning(
            "insight_generation_unexpected_error",
            error=str(exc)[:200],
        )

        return _fallback_insight(
            result=result,
            user_question=user_question,
        )


# =============================================================
# DETERMINISTIC DATA SUMMARY
# =============================================================

def _build_data_summary(
    df: pd.DataFrame,
    result: QueryResult,
) -> str:
    """
    Build a deterministic, factual summary for the insight LLM.

    The summary contains:
    - result size
    - column names
    - actual returned rows (when reasonably small)
    - numeric statistics
    - categorical value counts
    - explicitly detected outliers

    The LLM must use only this information when generating insights.
    """

    lines: list[str] = [
        f"Total rows: {result.row_count}"
        + (
            " (truncated — more rows existed)"
            if result.truncated
            else ""
        ),
        f"Columns: {', '.join(df.columns)}",
        "",
    ]

    # =========================================================
    # ACTUAL RESULT ROWS
    # =========================================================
    #
    # This is important.
    #
    # Previously the LLM only received statistics such as:
    #
    #   product_name: Chicken Wrap (1)
    #   category: Wrap (1)
    #
    # Now it can also see the actual returned record:
    #
    #   product_id=3 | product_name=Chicken Wrap |
    #   category=Wrap | price=8.50
    #
    # This gives the LLM factual context without giving it
    # access to the database itself.
    #
    if not df.empty:

        max_rows_for_context = 20

        rows_to_show = df.head(
            max_rows_for_context
        )

        lines.append("Returned rows:")

        for _, row in rows_to_show.iterrows():

            values = []

            for column in df.columns:

                value = row[column]

                if pd.isna(value):
                    value = "NULL"

                values.append(
                    f"{column}={value}"
                )

            lines.append(
                "  - " + " | ".join(values)
            )

        if len(df) > max_rows_for_context:

            lines.append(
                f"  ... {len(df) - max_rows_for_context} "
                "additional returned rows not shown"
            )

        lines.append("")

    # =========================================================
    # COLUMN TYPES
    # =========================================================

    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype.kind in ("i", "f", "u")
    ]

    categorical_cols = [
        c
        for c in df.columns
        if df[c].dtype.kind == "O"
    ]

    # =========================================================
    # NUMERIC STATISTICS
    # =========================================================

    if numeric_cols:

        lines.append(
            "Numeric column statistics:"
        )

        for col in numeric_cols:

            series = pd.to_numeric(
                df[col],
                errors="coerce",
            ).dropna()

            if series.empty:
                continue

            lines.append(
                f"  - {col}: "
                f"min={series.min():.2f}, "
                f"max={series.max():.2f}, "
                f"mean={series.mean():.2f}, "
                f"median={series.median():.2f}"
            )

        lines.append("")

    # =========================================================
    # OUTLIERS
    # =========================================================

    outlier_lines = _detect_outliers(
        df,
        numeric_cols,
    )

    if outlier_lines:

        lines.append(
            "Detected outliers "
            "(IQR method — values outside the typical range):"
        )

        lines.extend(
            f"  - {line}"
            for line in outlier_lines
        )

    else:

        lines.append(
            "Detected outliers: none."
        )

    lines.append("")

    # =========================================================
    # CATEGORICAL VALUES
    # =========================================================

    if categorical_cols:

        lines.append(
            "Categorical column value counts "
            "(top values):"
        )

        for col in categorical_cols:

            cardinality = df[col].nunique(
                dropna=True
            )

            if cardinality > 30:

                lines.append(
                    f"  - {col}: "
                    f"{cardinality} distinct values "
                    "(too many to list)"
                )

                continue

            counts = (
                df[col]
                .value_counts(dropna=False)
                .head(
                    _MAX_CATEGORICAL_VALUES_SHOWN
                )
            )

            formatted = ", ".join(
                f"{val} ({count})"
                for val, count in counts.items()
            )

            lines.append(
                f"  - {col}: {formatted}"
            )

    return "\n".join(lines)

# =============================================================
# OUTLIER DETECTION
# =============================================================

def _detect_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> list[str]:

    results: list[str] = []

    for col in numeric_cols:

        series = pd.to_numeric(
            df[col],
            errors="coerce",
        ).dropna()

        if len(series) < _MIN_ROWS_FOR_OUTLIER_CHECK:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series < lower_bound)
            | (series > upper_bound)
        ]

        if outliers.empty:
            continue

        examples = (
            outliers
            .sort_values(ascending=False)
            .head(_MAX_OUTLIER_EXAMPLES_PER_COLUMN)
        )

        example_str = ", ".join(
            f"{v:.2f}"
            for v in examples
        )

        results.append(
            f"{col}: "
            f"{len(outliers)} outlier value(s), "
            f"outside {lower_bound:.2f}–{upper_bound:.2f}. "
            f"Examples: {example_str}"
        )

    return results


# =============================================================
# RESPONSE PARSING
# =============================================================

def _parse_insight_response(
    raw_content: str,
) -> InsightResult | None:

    if not raw_content:
        return None

    cleaned = raw_content.strip()

    # Remove markdown code fences
    if cleaned.startswith("```"):

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

    try:

        data = json.loads(cleaned)

    except json.JSONDecodeError:

        # Try extracting JSON object
        match = re.search(
            r"\{.*\}",
            cleaned,
            re.DOTALL,
        )

        if not match:
            return None

        try:
            data = json.loads(match.group())

        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    return InsightResult(
        summary=str(
            data.get("summary", "")
        ).strip(),

        key_trends=_as_str_list(
            data.get("key_trends")
        ),

        outliers=_as_str_list(
            data.get("outliers")
        ),

        important_metrics=_as_str_list(
            data.get("important_metrics")
        ),

        follow_up_questions=_as_str_list(
            data.get("follow_up_questions")
        )[:3],
    )


def _as_str_list(
    value: object,
) -> list[str]:

    if value is None:
        return []

    if isinstance(value, str):

        value = value.strip()

        return [value] if value else []

    if isinstance(value, list):

        return [
            str(v).strip()
            for v in value
            if str(v).strip()
        ]

    return []


# =============================================================
# FALLBACKS
# =============================================================

def _build_basic_summary(
    result: QueryResult,
    user_question: str,
) -> str:

    return (
        f"The query returned {result.row_count} "
        f"row{'s' if result.row_count != 1 else ''} "
        f"for the requested analysis."
    )


def _default_follow_up_questions(
    result: QueryResult,
) -> list[str]:

    if result.row_count == 1:

        return [
            "Would you like more details about this result?",
            "Would you like to compare it with other records?",
            "Would you like to see related data?",
        ]

    return [
        "Would you like to compare these results?",
        "Would you like to see the results grouped by category?",
        "Would you like to analyze the key metrics?",
    ]


def _empty_result_insight(
    user_question: str,
) -> InsightResult:

    return InsightResult(
        summary="No data matched this query.",
        key_trends=[],
        outliers=[],
        important_metrics=[],
        follow_up_questions=[
            "Try broadening the filters in your question.",
            "Check whether the requested category or value exists.",
            "Ask a more general question to see what data is available.",
        ],
        is_empty=True,
    )


def _fallback_insight(
    result: QueryResult,
    user_question: str,
) -> InsightResult:

    return InsightResult(
        summary=_build_basic_summary(
            result,
            user_question,
        ),
        key_trends=[],
        outliers=[],
        important_metrics=[],
        follow_up_questions=_default_follow_up_questions(
            result
        ),
        is_empty=False,
    )