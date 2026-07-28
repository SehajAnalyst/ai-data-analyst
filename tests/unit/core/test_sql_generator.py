"""
tests/unit/core/test_sql_generator.py
========================================

Unit tests for core/nl2sql/sql_generator.py.

TESTING STRATEGY: FAKE LLM PROVIDER
---------------------------------------
These tests use a FakeLLMProvider that returns pre-scripted responses
instead of making network calls. This is the correct approach here for
two reasons:

  1. The logic being tested (retry loop, JSON parsing, schema
     consistency check, confidence derivation, conversation history
     formatting) is ALL deterministic Python code that doesn't
     involve the LLM at all — the LLM is a dependency to be mocked,
     not a thing being tested.
  2. Real API calls in unit tests mean: tests are slow, tests fail
     when the network is down, tests are non-deterministic, and tests
     cost money. None of that is acceptable for code that runs on
     every commit.

The integration between the generator and the REAL Groq API is tested
separately in tests/integration/ (once a real key is available in CI).

FAKE PROVIDER DESIGN: GroqProvider.generate() is mocked at the module
level by patching the factory's return value — this means
sql_generator.py's call to get_llm_provider() returns our fake, and
the fake's generate() is what actually runs. This matches how the
real pipeline works and avoids any coupling to GroqProvider internals.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from core.llm.base_provider import BaseLLMProvider, LLMResponse
from core.nl2sql.sql_generator import (
    _check_schema_consistency,
    _derive_confidence,
    _format_conversation_history,
    _parse_llm_response,
    generate_sql,
)
from core.schema.schema_context_builder import SchemaContext, build_schema_context
from core.schema.schema_introspector import introspect_schema
from exceptions.domain_exceptions import SQLGenerationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def employees_schema_context() -> SchemaContext:
    """Real SchemaContext built from an in-memory employees database."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text(
            """CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL,
                manager_id INTEGER,
                hire_date TEXT,
                FOREIGN KEY (manager_id) REFERENCES employees(id)
            )"""
        ))
        conn.execute(text(
            """CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget REAL
            )"""
        ))
        conn.commit()

    schema = introspect_schema(engine, include_row_counts=False)
    return build_schema_context(schema, "test question")


def make_fake_response(sql: str | None, reasoning: str, complexity_tier: str = "basic") -> LLMResponse:
    """Returns a fake LLMResponse containing a JSON payload that
    matches what the real Groq model is instructed to return."""
    payload = {
        "sql": sql,
        "reasoning": reasoning,
        "complexity_tier": complexity_tier,
    }
    return LLMResponse(
        content=json.dumps(payload),
        raw_provider_response=None,
        input_tokens=100,
        output_tokens=50,
        latency_ms=42.0,
        model="test-model",
    )


def make_fake_provider(responses: list[LLMResponse]) -> MagicMock:
    """Returns a mock that yields each response in order on successive
    generate() calls."""
    provider = MagicMock(spec=BaseLLMProvider)
    provider.provider_name = "fake"
    provider.generate.side_effect = responses
    return provider


# ---------------------------------------------------------------------------
# Tests: _parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def test_parses_clean_json(self) -> None:
        raw = json.dumps({
            "sql": "SELECT * FROM employees",
            "reasoning": "Simple lookup.",
            "complexity_tier": "basic",
        })
        result = _parse_llm_response(raw, "test question", 1)
        assert result is not None
        sql, reasoning, tier, clarification = result
        assert sql == "SELECT * FROM employees"
        assert tier == "basic"
        assert clarification is None

    def test_strips_markdown_code_fences(self) -> None:
        raw = "```json\n{\"sql\": \"SELECT 1\", \"reasoning\": \"test\", \"complexity_tier\": \"basic\"}\n```"
        result = _parse_llm_response(raw, "test", 1)
        assert result is not None
        assert result[0] == "SELECT 1"

    def test_null_sql_returns_clarification_request(self) -> None:
        raw = json.dumps({
            "sql": None,
            "reasoning": "Which department do you mean?",
            "complexity_tier": None,
        })
        result = _parse_llm_response(raw, "test", 1)
        assert result is not None
        sql, reasoning, tier, clarification = result
        assert sql is None
        assert clarification == "Which department do you mean?"

    def test_invalid_json_returns_none(self) -> None:
        result = _parse_llm_response("this is not json at all", "test", 1)
        assert result is None

    def test_json_embedded_in_prose_is_extracted(self) -> None:
        raw = 'Sure! Here is the SQL: {"sql": "SELECT id FROM employees", "reasoning": "r", "complexity_tier": "basic"} Hope that helps!'
        result = _parse_llm_response(raw, "test", 1)
        assert result is not None
        assert result[0] == "SELECT id FROM employees"

    def test_unknown_complexity_tier_normalised_to_basic(self) -> None:
        raw = json.dumps({"sql": "SELECT 1", "reasoning": "r", "complexity_tier": "medium"})
        result = _parse_llm_response(raw, "test", 1)
        assert result[2] == "basic"


# ---------------------------------------------------------------------------
# Tests: _check_schema_consistency
# ---------------------------------------------------------------------------


class TestCheckSchemaConsistency:
    def test_valid_table_passes(self, employees_schema_context: SchemaContext) -> None:
        sql = "SELECT name, salary FROM employees WHERE salary > 50000"
        error = _check_schema_consistency(sql, employees_schema_context, "I used employees table")
        assert error == ""

    def test_hallucinated_table_caught(self, employees_schema_context: SchemaContext) -> None:
        sql = "SELECT * FROM payroll WHERE salary > 50000"
        error = _check_schema_consistency(sql, employees_schema_context, "used payroll")
        assert "payroll" in error
        assert "employees" in error  # error message includes valid tables

    def test_hallucinated_qualified_column_caught(self, employees_schema_context: SchemaContext) -> None:
        sql = "SELECT employees.reveune FROM employees"
        error = _check_schema_consistency(sql, employees_schema_context, "used reveune")
        assert "reveune" in error

    def test_valid_join_passes(self, employees_schema_context: SchemaContext) -> None:
        sql = "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department = d.name"
        error = _check_schema_consistency(sql, employees_schema_context, "joined tables")
        assert error == ""

    def test_empty_schema_always_passes(self) -> None:
        empty_schema = SchemaContext(
            relevant_tables=[],
            formatted_text="This database has no tables.",
            retrieval_method="full_schema",
        )
        error = _check_schema_consistency("SELECT 1", empty_schema, "no schema")
        assert error == ""


# ---------------------------------------------------------------------------
# Tests: _derive_confidence
# ---------------------------------------------------------------------------


class TestDeriveConfidence:
    def test_first_attempt_clean_question_is_high(self, employees_schema_context: SchemaContext) -> None:
        score = _derive_confidence("Used employees table for salary lookup.", 1, "show all employees", employees_schema_context)
        assert score >= 0.8

    def test_second_attempt_lower_than_first(self, employees_schema_context: SchemaContext) -> None:
        score1 = _derive_confidence("clean", 1, "show salary", employees_schema_context)
        score2 = _derive_confidence("clean", 2, "show salary", employees_schema_context)
        assert score2 < score1

    def test_uncertainty_language_lowers_score(self, employees_schema_context: SchemaContext) -> None:
        clean_score = _derive_confidence("Used employees table.", 1, "show salary", employees_schema_context)
        uncertain_score = _derive_confidence("I think I used employees table, but I'm not sure.", 1, "show salary", employees_schema_context)
        assert uncertain_score < clean_score

    def test_score_clamps_between_0_and_1(self, employees_schema_context: SchemaContext) -> None:
        score = _derive_confidence("uncertain, assume, might, could, unclear", 5, "foo bar baz", employees_schema_context)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests: _format_conversation_history
# ---------------------------------------------------------------------------


class TestFormatConversationHistory:
    def test_none_returns_empty_string(self) -> None:
        assert _format_conversation_history(None) == ""

    def test_empty_list_returns_empty_string(self) -> None:
        assert _format_conversation_history([]) == ""

    def test_formats_user_and_assistant_turns(self) -> None:
        history = [
            {"role": "user", "content": "How many employees?"},
            {"role": "assistant", "content": {"sql": "SELECT COUNT(*) FROM employees"}},
        ]
        result = _format_conversation_history(history)
        assert "How many employees?" in result
        assert "SELECT COUNT(*) FROM employees" in result

    def test_caps_at_six_turns(self) -> None:
        history = [{"role": "user", "content": f"Q{i}"} for i in range(10)]
        result = _format_conversation_history(history)
        assert "Q9" in result   # most recent included
        assert "Q0" not in result  # oldest dropped


# ---------------------------------------------------------------------------
# Tests: generate_sql (integration-ish, but no real API calls)
# ---------------------------------------------------------------------------


class TestGenerateSQL:
    def test_successful_first_attempt(self, employees_schema_context: SchemaContext) -> None:
        fake_response = make_fake_response(
            sql="SELECT name, salary FROM employees WHERE salary > 80000 LIMIT 100",
            reasoning="Used employees table, filtered by salary.",
            complexity_tier="basic",
        )
        with patch("core.nl2sql.sql_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = make_fake_provider([fake_response])
            result = generate_sql("Show employees earning more than 80000", employees_schema_context)

        assert result.sql == "SELECT name, salary FROM employees WHERE salary > 80000 LIMIT 100"
        assert result.attempt_count == 1
        assert result.clarification_request is None
        assert result.confidence > 0.5
        assert result.question == "Show employees earning more than 80000"

    def test_retry_on_schema_inconsistency(self, employees_schema_context: SchemaContext) -> None:
        bad_response = make_fake_response(
            sql="SELECT * FROM payroll WHERE salary > 80000",  # hallucinated table
            reasoning="Used payroll table.",
        )
        good_response = make_fake_response(
            sql="SELECT * FROM employees WHERE salary > 80000 LIMIT 100",
            reasoning="Used employees table.",
        )
        with patch("core.nl2sql.sql_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = make_fake_provider([bad_response, good_response])
            result = generate_sql("Show employees earning more than 80000", employees_schema_context)

        assert result.attempt_count == 2
        assert "employees" in result.sql

    def test_retry_on_bad_json(self, employees_schema_context: SchemaContext) -> None:
        bad_json_response = LLMResponse(
            content="I'll generate the SQL for you: SELECT * FROM employees",
            raw_provider_response=None,
            input_tokens=50, output_tokens=30, latency_ms=10.0, model="test",
        )
        good_response = make_fake_response(
            sql="SELECT * FROM employees LIMIT 100",
            reasoning="Used employees.",
        )
        with patch("core.nl2sql.sql_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = make_fake_provider([bad_json_response, good_response])
            result = generate_sql("Show all employees", employees_schema_context)

        assert result.attempt_count == 2

    def test_raises_after_all_retries_exhausted(self, employees_schema_context: SchemaContext) -> None:
        bad_response = make_fake_response(
            sql="SELECT * FROM nonexistent_table",
            reasoning="Used nonexistent_table.",
        )
        # Return the bad response for every attempt.
        with patch("core.nl2sql.sql_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = make_fake_provider([bad_response] * 10)
            with pytest.raises(SQLGenerationError) as exc_info:
                generate_sql("Show all employees", employees_schema_context)

        assert exc_info.value.user_message  # has a safe user-facing message

    def test_clarification_request_returned_not_raised(self, employees_schema_context: SchemaContext) -> None:
        clarification_response = make_fake_response(
            sql=None,
            reasoning="Which department do you mean? There are multiple departments.",
        )
        with patch("core.nl2sql.sql_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = make_fake_provider([clarification_response])
            result = generate_sql("Show the sales", employees_schema_context)

        assert result.sql is None
        assert result.clarification_request is not None
        assert "department" in result.clarification_request.lower()
        assert result.confidence == 0.4

    def test_empty_question_raises(self, employees_schema_context: SchemaContext) -> None:
        with pytest.raises(SQLGenerationError):
            generate_sql("   ", employees_schema_context)

    def test_conversation_history_passed_to_prompt(self, employees_schema_context: SchemaContext) -> None:
        history = [{"role": "user", "content": "How many employees?"}]
        good_response = make_fake_response(
            sql="SELECT COUNT(*) FROM employees",
            reasoning="Count query.",
        )
        captured_calls = []

        def capture_generate(system_prompt, user_message, **kwargs):
            captured_calls.append({"system": system_prompt, "user": user_message})
            return good_response

        with patch("core.nl2sql.sql_generator.get_llm_provider") as mock_factory:
            fake = MagicMock(spec=BaseLLMProvider)
            fake.generate.side_effect = capture_generate
            mock_factory.return_value = fake
            generate_sql("Now show by department", employees_schema_context, conversation_history=history)

        assert len(captured_calls) == 1
        assert "How many employees?" in captured_calls[0]["system"]
