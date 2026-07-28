"""
tests/unit/core/test_insight_generator.py
============================================

Tests for core/insights/insight_generator.py.

Uses a fake LLM provider (same pattern as test_sql_generator.py) so
these tests are fast, deterministic, and don't require a real Groq
API key. The deterministic stats/outlier computation is tested
directly since it's the part that must be correct independent of
any LLM behavior.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from core.execution.query_executor import QueryResult
from core.insights.insight_generator import (
    _build_data_summary,
    _detect_outliers,
    _parse_insight_response,
    generate_insight,
)
from exceptions.domain_exceptions import LLMAPIError, LLMTimeoutError


def make_result(data: dict, truncated: bool = False) -> QueryResult:
    df = pd.DataFrame(data)
    return QueryResult(
        columns=list(df.columns),
        rows=[tuple(r) for r in df.values],
        row_count=len(df),
        truncated=truncated,
        execution_time_ms=5.0,
    )


def make_fake_provider(response_content: str) -> MagicMock:
    fake_response = MagicMock()
    fake_response.content = response_content
    provider = MagicMock()
    provider.generate.return_value = fake_response
    return provider


# ── Deterministic stats computation ────────────────────────────────────────


class TestDataSummary:
    def test_includes_row_and_column_counts(self):
        result = make_result({"a": [1, 2, 3]})
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = _build_data_summary(df, result)
        assert "Total rows: 3" in summary
        assert "Columns: a" in summary

    def test_truncation_noted(self):
        result = make_result({"a": [1, 2, 3]}, truncated=True)
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = _build_data_summary(df, result)
        assert "truncated" in summary.lower()

    def test_numeric_stats_present(self):
        df = pd.DataFrame({"salary": [50000.0, 60000.0, 70000.0]})
        result = make_result({"salary": [50000.0, 60000.0, 70000.0]})
        summary = _build_data_summary(df, result)
        assert "min=50000.00" in summary
        assert "max=70000.00" in summary

    def test_categorical_value_counts_present(self):
        df = pd.DataFrame({"dept": ["Eng", "Eng", "Mkt"]})
        result = make_result({"dept": ["Eng", "Eng", "Mkt"]})
        summary = _build_data_summary(df, result)
        assert "Eng (2)" in summary
        assert "Mkt (1)" in summary

    def test_high_cardinality_categorical_not_enumerated(self):
        values = [f"item_{i}" for i in range(50)]
        df = pd.DataFrame({"name": values})
        result = make_result({"name": values})
        summary = _build_data_summary(df, result)
        assert "too many to list" in summary

    def test_no_outliers_reported_when_none_exist(self):
        df = pd.DataFrame({"salary": [50000.0, 51000.0, 52000.0, 53000.0]})
        result = make_result({"salary": [50000.0, 51000.0, 52000.0, 53000.0]})
        summary = _build_data_summary(df, result)
        assert "Detected outliers: none." in summary


class TestOutlierDetection:
    def test_detects_planted_outlier(self):
        df = pd.DataFrame({"salary": [90000.0, 95000.0, 250000.0, 70000.0, 72000.0, 61000.0]})
        outliers = _detect_outliers(df, ["salary"])
        assert len(outliers) == 1
        assert "250000" in outliers[0]

    def test_no_outliers_in_uniform_data(self):
        df = pd.DataFrame({"salary": [50000.0, 51000.0, 52000.0, 53000.0, 54000.0]})
        outliers = _detect_outliers(df, ["salary"])
        assert outliers == []

    def test_skips_columns_below_minimum_sample_size(self):
        df = pd.DataFrame({"salary": [50000.0, 999999.0]})  # only 2 rows
        outliers = _detect_outliers(df, ["salary"])
        assert outliers == []  # too few rows for meaningful IQR

    def test_identical_values_no_outliers(self):
        df = pd.DataFrame({"salary": [50000.0] * 10})
        outliers = _detect_outliers(df, ["salary"])
        assert outliers == []  # IQR is 0, guarded against division issues


# ── JSON response parsing ──────────────────────────────────────────────────


class TestParseInsightResponse:
    def test_parses_clean_json(self):
        raw = json.dumps({
            "summary": "Test",
            "key_trends": ["trend1"],
            "outliers": [],
            "important_metrics": ["metric1"],
            "follow_up_questions": ["q1", "q2", "q3"],
        })
        result = _parse_insight_response(raw)
        assert result is not None
        assert result.summary == "Test"
        assert result.key_trends == ["trend1"]

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps({
            "summary": "fenced", "key_trends": [], "outliers": [],
            "important_metrics": [], "follow_up_questions": [],
        }) + "\n```"
        result = _parse_insight_response(raw)
        assert result is not None
        assert result.summary == "fenced"

    def test_extracts_json_from_surrounding_prose(self):
        raw = 'Here is the analysis: ' + json.dumps({
            "summary": "extracted", "key_trends": [], "outliers": [],
            "important_metrics": [], "follow_up_questions": [],
        }) + ' Hope this helps!'
        result = _parse_insight_response(raw)
        assert result is not None
        assert result.summary == "extracted"

    def test_invalid_json_returns_none(self):
        assert _parse_insight_response("not json") is None

    def test_follow_up_questions_capped_at_three(self):
        raw = json.dumps({
            "summary": "s", "key_trends": [], "outliers": [],
            "important_metrics": [], "follow_up_questions": ["q1", "q2", "q3", "q4", "q5"],
        })
        result = _parse_insight_response(raw)
        assert len(result.follow_up_questions) == 3

    def test_tolerates_string_instead_of_list(self):
        """Model sometimes returns a bare string for a field that
        should be a list — must not crash."""
        raw = json.dumps({
            "summary": "s", "key_trends": "single trend as string",
            "outliers": [], "important_metrics": [], "follow_up_questions": [],
        })
        result = _parse_insight_response(raw)
        assert result is not None
        assert result.key_trends == ["single trend as string"]


# ── generate_insight orchestration ─────────────────────────────────────────


class TestGenerateInsight:
    def test_empty_result_skips_llm_call(self):
        empty = QueryResult(columns=["x"], rows=[], row_count=0,
                            truncated=False, execution_time_ms=1.0)
        with patch("core.insights.insight_generator.get_llm_provider") as mock_factory:
            insight = generate_insight("q", "SELECT 1", empty)
            mock_factory.assert_not_called()

        assert insight.is_empty is True
        assert insight.summary == "No data matched this query."
        assert len(insight.follow_up_questions) == 3

    def test_successful_generation(self):
        result = make_result({"dept": ["Eng", "Mkt"], "salary": [90000.0, 70000.0]})
        fake_content = json.dumps({
            "summary": "Engineering earns more than Marketing.",
            "key_trends": ["Eng salary exceeds Mkt"],
            "outliers": [],
            "important_metrics": ["Average salary is 80000"],
            "follow_up_questions": ["What about HR?", "Trend over time?", "By seniority?"],
        })
        with patch("core.insights.insight_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = make_fake_provider(fake_content)
            insight = generate_insight("avg salary by dept", "SELECT * FROM employees", result)

        assert insight.summary == "Engineering earns more than Marketing."
        assert insight.is_empty is False
        assert len(insight.follow_up_questions) == 3

    def test_llm_api_error_returns_fallback_not_raise(self):
        result = make_result({"salary": [50000.0, 60000.0]})
        with patch("core.insights.insight_generator.get_llm_provider") as mock_factory:
            provider = MagicMock()
            provider.generate.side_effect = LLMAPIError("boom", user_message="fail")
            mock_factory.return_value = provider
            insight = generate_insight("q", "SELECT 1", result)  # must not raise

        assert insight.summary
        assert insight.key_trends == []

    def test_llm_timeout_returns_fallback_not_raise(self):
        result = make_result({"salary": [50000.0, 60000.0]})
        with patch("core.insights.insight_generator.get_llm_provider") as mock_factory:
            provider = MagicMock()
            provider.generate.side_effect = LLMTimeoutError("timeout", user_message="slow")
            mock_factory.return_value = provider
            insight = generate_insight("q", "SELECT 1", result)  # must not raise

        assert insight.summary

    def test_unparseable_response_returns_fallback(self):
        result = make_result({"salary": [50000.0, 60000.0]})
        with patch("core.insights.insight_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = make_fake_provider("garbage not json")
            insight = generate_insight("q", "SELECT 1", result)

        assert insight.summary
        assert insight.key_trends == []
