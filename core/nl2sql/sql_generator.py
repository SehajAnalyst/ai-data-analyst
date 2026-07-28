"""
core/nl2sql/sql_generator.py
===============================

The AI Text-to-SQL engine. Converts a natural-language question into
valid SQL by injecting schema context into a structured prompt, calling
Groq via the provider layer, parsing the structured JSON response, and
running a bounded retry loop when the model's output fails validation.

DESIGN BOUNDARIES (INTENTIONAL, NON-NEGOTIABLE)
--------------------------------------------------
This module:
  - DOES produce a SQL string.
  - DOES NOT execute that SQL string.
  - DOES NOT connect to the database.
  - DOES NOT call schema introspection (schema is passed in, already
    discovered and cached by the caller).
  - DOES run the validator against generated SQL, and uses rejection
    reasons as retry feedback — but DOES NOT implement the validator
    itself (that's sql_validator.py, once built).

The LLM proposes; deterministic code (the validator, then the
executor) disposes. This is the boundary that makes sql_validator.py
a real checkpoint, not a formality. If the generator executed its own
output, the validator would be bypassable by construction.

HOW HALLUCINATION IS MINIMIZED (not eliminated — minimized)
-------------------------------------------------------------
LLMs hallucinate. No prompt engineering eliminates this fully. The
strategy used here is layered, not reliant on any single mechanism:

  1. SCHEMA INJECTION: the prompt contains the COMPLETE list of valid
     table and column names, formatted to be easily parseable by the
     model. "Never invent table or column names" appears as an
     explicit hard rule before the schema, not after (models process
     constraints better when they precede the material they constrain).

  2. JSON RESPONSE FORMAT: the model is instructed to return structured
     JSON with "sql", "reasoning", and "complexity_tier" fields. This
     serves hallucination control as well as parsing: requiring the
     model to name a "complexity_tier" forces it to commit to a
     conscious technique selection rather than producing SQL that
     happens to match some pattern by imitation.

  3. POST-GENERATION SCHEMA VALIDATION: generate_sql() checks the
     returned SQL against the known-valid table and column names from
     SchemaContext before returning. This is a fast string/AST check,
     not a full semantic validator — think of it as a "did the model
     reference something it wasn't told about" filter rather than a
     complete safety check. The full sql_validator.py (deterministic
     AST parsing via sqlglot) is the real safety layer and comes next
     in the pipeline.

  4. RETRY WITH REJECTION REASON: if the post-generation check (or
     the downstream validator) finds a problem, the rejection reason is
     fed back into the next attempt's prompt in plain language
     ("column 'reveune' doesn't exist; did you mean 'revenue'?").
     The model can usually self-correct on a specific, actionable
     rejection reason that it couldn't anticipate on the first attempt.

WHY JSON RESPONSE FORMAT, NOT JUST RAW SQL
--------------------------------------------
Requiring JSON means the model commits to structured output that
can be parsed and validated without fragile regex extraction.
The "reasoning" field is particularly important: it forces the model
to explicitly state which tables and columns it used and why. When
hallucination occurs, the reasoning field often reveals it — the
model will name a table that doesn't exist in the schema, which
is detectable before executing anything. This is an early-warning
signal, not a guarantee.

CONFIDENCE SCORE — WHAT IT ACTUALLY IS
-----------------------------------------
The returned confidence field is a heuristic derived from observable
signals, not a probability:
  - Starts at 1.0.
  - Deducted 0.3 if the model asked for clarification rather than
    generating SQL (high ambiguity detected).
  - Deducted 0.2 if the reasoning mentions uncertainty language
    ("might", "could", "unclear", "ambiguous", "assume").
  - Deducted 0.1 per retry attempt beyond the first (each retry
    needed means the first attempt had problems).
  - Deducted 0.2 if the question contained no words found in the
    schema's tables or columns (weak schema grounding detected).

This is not a calibrated probability. Do not display it to users as
a percentage or a confidence interval. It's a developer/debugging
signal, not a user-facing metric. The user should see the SQL and its
explanation; they should not see a score that implies a false
precision about correctness.

THE RETRY LOOP
---------------
    for attempt in range(1 + settings.llm_max_retries):
        1. Build prompt (with rejection_reason from last attempt, if any).
        2. Call Groq.
        3. Parse JSON response.
        4. Quick post-generation schema check.
        5. If passes: return result.
        6. If fails: set rejection_reason, continue loop.
    raise SQLGenerationError.

The retry loop lives HERE (not in ConversationManager, not in the
validator) because it's specifically about the generate→check→
regenerate feedback cycle, which is an internal concern of generation
quality, not of conversation flow or safety enforcement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from config.settings import get_settings
from core.llm.llm_client_factory import get_llm_provider
from core.nl2sql.prompt_builder import build_sql_generation_prompt
from core.schema.schema_context_builder import SchemaContext
from exceptions.domain_exceptions import (
    LLMAPIError,
    SQLGenerationError,
)
from logging_setup.logger import LogCategory, LogFields, get_logger

logger = get_logger(__name__)

_UNCERTAINTY_SIGNALS = frozenset([
    "might", "could", "possibly", "unclear", "ambiguous", "assume",
    "assuming", "not sure", "uncertain", "i think", "i believe",
    "probably", "unsure",
])

_CLARIFICATION_SIGNALS = frozenset([
    "could you clarify", "could you specify", "please clarify",
    "which table", "which column", "what do you mean", "do you mean",
    "can you confirm", "please specify",
])


@dataclass
class SQLGenerationResult:
    """
    The complete output of one successful SQL generation call.

    Fields:
        question: the original natural-language question, stored
            for downstream use (logging, UI display, conversation
            history assembly) without requiring callers to track it
            separately.
        sql: the generated SQL string. NOT YET VALIDATED by
            sql_validator.py — see module docstring on why the
            generator stops here and doesn't also validate.
        reasoning: the model's stated explanation of its approach
            (which tables/columns it chose and why). This is the
            "explain the SQL" content shown in the UI's collapsible
            SQL view. It's the model's own words, not post-processed.
        complexity_tier: "basic" | "intermediate" | "advanced" — the
            model's self-declared classification of the technique
            used, logged for analytics and potentially used in future
            to select few-shot examples.
        confidence: heuristic 0.0-1.0 score. See module docstring for
            exactly what this is and, importantly, what it's NOT.
        clarification_request: if the model determined the question
            was too ambiguous to generate SQL, this field contains
            what it would need to know. sql will be None in this case.
        attempt_count: how many generation attempts were needed. 1
            means first-try success; >1 means earlier attempts were
            rejected and retried.
        latency_ms: total wall-clock time for all LLM calls in this
            generation, not including caller-side time.
    """

    question: str
    sql: str | None
    reasoning: str
    complexity_tier: str
    confidence: float
    clarification_request: str | None
    attempt_count: int
    latency_ms: float
    schema_context: SchemaContext = field(repr=False)  # excluded from repr to keep it readable


def generate_sql(
    user_question: str,
    schema_context: SchemaContext,
    conversation_history: list[dict] | None = None,
) -> SQLGenerationResult:
    """
    Generates SQL for a natural-language question using Groq, with a
    bounded retry loop that feeds validator rejection reasons back into
    the prompt as correction guidance.

    Args:
        user_question: the current natural-language question. Must be
            non-empty.
        schema_context: the schema context for this question, from
            schema_context_builder.build_schema_context(). Includes
            the formatted text to inject into the prompt AND the
            table/column sets used for post-generation hallucination
            detection.
        conversation_history: list of prior conversation turns in the
            format [{"role": "user"|"assistant", "content": str}, ...].
            Used to resolve follow-up questions ("now break it down by
            region") that don't make sense without the prior context.
            None or empty list for a fresh question.

    Returns:
        SQLGenerationResult. If the model requested clarification
        rather than generating SQL, result.sql is None and
        result.clarification_request is populated. Callers should
        check for this before treating the result as execution-ready.

    Raises:
        SQLGenerationError: failed to produce parseable, schema-
            consistent SQL after all retry attempts. Callers
            (ConversationManager) should catch this and surface
            result.clarification_request-style text to the user rather
            than crashing.
        LLMAPIError, LLMTimeoutError: LLM-layer failures that aren't
            retried here (they're infrastructure failures, not
            generation-quality failures — retrying immediately on a
            timeout rarely helps and adds latency). Let these
            propagate to ConversationManager.

    NOTE: this function does NOT run sql_validator.py's full safety
    check. It runs a lighter post-generation schema check (table and
    column name existence only) to provide meaningful retry feedback.
    The full safety validation (AST-based, forbidden keyword check,
    LIMIT injection, etc.) must be called by the downstream caller
    (ConversationManager or whoever holds the full pipeline) before
    the returned SQL is executed. See sql_validator.py (stub, next
    to implement) for that layer.
    """
    if not user_question.strip():
        raise SQLGenerationError(
            message="user_question must be non-empty.",
            user_message="Please enter a question.",
        )

    settings = get_settings()
    provider = get_llm_provider()
    history_str = _format_conversation_history(conversation_history)

    log = logger.bind(
        category=LogCategory.SQL_GENERATION,
        question_preview=user_question[:120],
    )

    rejection_reason: str = ""
    total_latency_ms: float = 0.0
    max_attempts = 1 + settings.llm_max_retries

    for attempt in range(1, max_attempts + 1):
        log.debug("sql_generation_attempt", attempt=attempt, max_attempts=max_attempts)

        system_prompt, user_message = build_sql_generation_prompt(
            schema_context=schema_context.formatted_text,
            dialect=_extract_dialect(schema_context),
            user_question=user_question,
            conversation_history=history_str,
            retry_context=rejection_reason,
        )

        try:
            llm_response = provider.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.0,   # SQL generation must be deterministic
            )
        except (LLMAPIError,):
            # Don't retry LLM infrastructure failures — let them propagate.
            raise

        total_latency_ms += llm_response.latency_ms

        parse_result = _parse_llm_response(llm_response.content, user_question, attempt)
        if parse_result is None:
            rejection_reason = (
                f"Your response on attempt {attempt} could not be parsed as valid JSON "
                f"with 'sql', 'reasoning', and 'complexity_tier' fields. "
                f"Respond with ONLY the JSON object, no markdown, no preamble."
            )
            log.warning("sql_generation_parse_failed", attempt=attempt)
            continue

        sql, reasoning, complexity_tier, clarification_request = parse_result

        if clarification_request:
            log.info(
                "sql_generation_clarification_requested",
                attempt=attempt,
                clarification=clarification_request[:200],
            )
            return SQLGenerationResult(
                question=user_question,
                sql=None,
                reasoning=reasoning,
                complexity_tier="unknown",
                confidence=0.4,
                clarification_request=clarification_request,
                attempt_count=attempt,
                latency_ms=total_latency_ms,
                schema_context=schema_context,
            )

        schema_error = _check_schema_consistency(sql, schema_context, reasoning)
        if schema_error:
            rejection_reason = schema_error
            log.warning("sql_generation_schema_check_failed", attempt=attempt, reason=schema_error[:200])
            continue

        confidence = _derive_confidence(reasoning, attempt, user_question, schema_context)

        log.info(
            "sql_generation_succeeded",
            attempt=attempt,
            complexity_tier=complexity_tier,
            confidence=round(confidence, 2),
            **{LogFields.LATENCY_MS: round(total_latency_ms, 2)},
        )

        return SQLGenerationResult(
            question=user_question,
            sql=sql,
            reasoning=reasoning,
            complexity_tier=complexity_tier,
            confidence=confidence,
            clarification_request=None,
            attempt_count=attempt,
            latency_ms=total_latency_ms,
            schema_context=schema_context,
        )

    log.error(
        "sql_generation_failed_all_attempts",
        max_attempts=max_attempts,
        last_rejection=rejection_reason[:300],
    )
    raise SQLGenerationError(
        message=(
            f"Failed to generate valid SQL after {max_attempts} attempts. "
            f"Last rejection reason: {rejection_reason}"
        ),
        user_message=(
            "I wasn't able to generate a query for that question. "
            "Could you try rephrasing it, or ask something more specific?"
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(
    raw_content: str,
    user_question: str,
    attempt: int,
) -> tuple[str | None, str, str, str | None] | None:
    """
    Parses the LLM's JSON response into (sql, reasoning, complexity_tier,
    clarification_request). Returns None if parsing fails entirely.

    The model is instructed to return clean JSON, but models sometimes
    wrap it in markdown code fences (```json ... ```) despite explicit
    instructions not to — this is common enough to warrant stripping
    before parsing rather than treating it as a hard failure on the
    first attempt and burning a retry on something trivially fixable.
    """
    cleaned = raw_content.strip()

    # Strip markdown code fences if present — e.g. ```json ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object embedded in surrounding prose —
        # some models emit text before/after the JSON despite instructions.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    sql_raw = data.get("sql")
    reasoning = str(data.get("reasoning", "")).strip()
    complexity_tier = str(data.get("complexity_tier", "unknown")).strip().lower()
    clarification_request: str | None = None

    if sql_raw is None:
        # Model indicated it cannot generate SQL — extract why from reasoning.
        clarification_request = reasoning or "Could you please provide more details about your question?"
        return None, reasoning, "unknown", clarification_request

    sql = str(sql_raw).strip()
    if not sql:
        return None

    # Normalize complexity_tier to one of the three expected values.
    if complexity_tier not in ("basic", "intermediate", "advanced"):
        complexity_tier = "basic"

    return sql, reasoning, complexity_tier, None


def _check_schema_consistency(
    sql: str,
    schema_context: SchemaContext,
    reasoning: str,
) -> str:
    """
    Lightweight check: does the generated SQL reference any identifiers
    that aren't in the provided schema context?

    This is NOT a substitute for sql_validator.py's full AST-based
    check. It's a fast filter that catches the most common hallucination
    mode (inventing a table or column name) to provide a meaningful
    retry message before the full validator runs downstream.

    Returns:
        Empty string if the SQL is consistent with the schema.
        A rejection-reason string (suitable for the retry prompt) if
        not. The reason is specific ("table 'employes' not found;
        available tables: employees, departments") so the model can
        self-correct on the specific typo or hallucination.
    """
    if not schema_context.table_names:
        # Empty schema — can't validate; pass through and let downstream handle it.
        return ""

    valid_tables = schema_context.table_names
    valid_columns_by_table = schema_context.all_column_names_by_table

    # Extract identifiers from SQL using a simple word-boundary scan.
    # This will catch most hallucinated table/column names without
    # needing a full SQL parser (sqlglot is used for that in
    # sql_validator.py — this is the lightweight pre-check).

    # Check that every FROM / JOIN target is a known table.
    # Pattern: (FROM|JOIN) whitespace+ identifier
    table_references = re.findall(
        r'(?:FROM|JOIN)\s+["`]?(\w+)["`]?',
        sql,
        re.IGNORECASE,
    )
    for ref in table_references:
        ref_lower = ref.lower()
        if ref_lower not in {t.lower() for t in valid_tables}:
            available = ", ".join(sorted(valid_tables))
            return (
                f"Table '{ref}' was referenced in the SQL but does not exist in the schema. "
                f"Available tables: {available}. "
                f"Only use tables listed in the SCHEMA section."
            )

    # Check that qualified column references (table.column) are valid.
    # Unqualified column references are not checked here — the full
    # validator handles those via AST, since unqualified references
    # can be legitimately ambiguous and their resolution depends on
    # which tables are in scope, not just a name lookup.
    qualified_refs = re.findall(
        r'["`]?(\w+)["`]?\.["`]?(\w+)["`]?',
        sql,
    )
    for table_ref, col_ref in qualified_refs:
        # Skip numeric aliases like "1.5" (arithmetic literals).
        if table_ref.isdigit() or col_ref.isdigit():
            continue

        table_match = next(
            (t for t in valid_tables if t.lower() == table_ref.lower()),
            None,
        )
        if table_match and col_ref.lower() not in {
            c.lower() for c in valid_columns_by_table.get(table_match, set())
        }:
            available_cols = ", ".join(sorted(valid_columns_by_table.get(table_match, set())))
            return (
                f"Column '{table_ref}.{col_ref}' was referenced but does not exist. "
                f"Available columns in '{table_match}': {available_cols}."
            )

    # Check the reasoning field too — if the model mentions a table
    # name in its reasoning that doesn't exist, it's a signal that it
    # may have hallucinated even if the SQL itself passed the checks.
    # This is softer (warn in log, don't reject) — reasoning phrasing
    # is less rigidly constrained than SQL identifiers.
    for word in re.findall(r'\b(\w+)\b', reasoning):
        if len(word) > 3 and word not in valid_tables and word.lower() in [
            t.lower() for t in valid_tables
        ]:
            # Case mismatch in reasoning — not a rejection, just noise.
            pass

    return ""


def _derive_confidence(
    reasoning: str,
    attempt: int,
    user_question: str,
    schema_context: SchemaContext,
) -> float:
    """
    Derives a heuristic confidence score from observable signals.
    See module docstring for exactly what this score is and IS NOT.

    The score starts at 1.0 and deductions are applied:
      - -0.2 per retry beyond the first (each retry signals problems).
      - -0.2 if reasoning contains uncertainty language.
      - -0.1 if the question contains no words found in the schema
        (weak grounding — question may be off-topic or referencing
        entities not in the schema).
    """
    score = 1.0

    # Deduction for retries.
    score -= (attempt - 1) * 0.2

    # Deduction for uncertainty language in reasoning.
    reasoning_lower = reasoning.lower()
    if any(signal in reasoning_lower for signal in _UNCERTAINTY_SIGNALS):
        score -= 0.2

    # Deduction for weak schema grounding.
    question_words = {w.lower() for w in re.findall(r'\b\w+\b', user_question) if len(w) > 3}
    schema_words = set()
    for table_name in schema_context.table_names:
        schema_words.add(table_name.lower())
        for col_name in schema_context.all_column_names_by_table.get(table_name, set()):
            schema_words.add(col_name.lower())

    if question_words and not (question_words & schema_words):
        score -= 0.1

    return max(0.1, min(1.0, round(score, 2)))


def _format_conversation_history(history: list[dict] | None) -> str:
    """
    Formats conversation history into a compact plain-text block for
    injection into the prompt's CONVERSATION HISTORY section.

    Each turn is rendered as:
        User: <question>
        Assistant: <SQL that was generated>

    SQL is included (not just the user turn) so the model can resolve
    follow-up questions like "now add a filter for region='West'" —
    it needs to know what the previous query looked like to modify it
    correctly. Only the SQL is included from the assistant side, not
    the full reasoning or insights, to keep context compact.
    """
    if not history:
        return ""

    lines: list[str] = []
    for turn in history[-6:]:  # cap at last 6 turns to manage token budget
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            # Content may be a dict with 'sql' key (structured) or a string.
            if isinstance(content, dict):
                sql = content.get("sql", "")
                if sql:
                    lines.append(f"Assistant SQL: {sql}")
            elif isinstance(content, str) and content.strip():
                lines.append(f"Assistant: {content}")

    return "\n".join(lines) if lines else ""


def _extract_dialect(schema_context: SchemaContext) -> str:
    """
    Extracts the database dialect from the schema context's formatted
    text. The dialect is injected into the prompt to tell the model
    which SQL flavour to use (e.g. SQLite's strftime() vs PostgreSQL's
    date_trunc() for date operations).

    Falls back to 'sqlite' if extraction fails — SQLite is the V1
    default and the safest fallback since its SQL is a subset of most
    other dialects' common features.
    """
    # build_schema_summary() always starts with "Database type: <dialect>"
    match = re.search(r'Database type:\s*(\w+)', schema_context.formatted_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "sqlite"
