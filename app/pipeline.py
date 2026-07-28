"""
app/pipeline.py
=================

Orchestrates the full question → SQL → validate → execute → result
workflow for one user turn.

WHY THIS LIVES IN app/ NOT core/
----------------------------------
This module reads from app/state/session_state to get the engine and
schema, then calls core/ modules in sequence. It has Streamlit state
dependencies (via session_state) which means it belongs in the app
layer, not in core/ which must stay Streamlit-free.

The pipeline function is the only place in the app that calls
sql_generator, sql_validator, and query_executor together. Every error
is caught and converted into a ChatMessage with an appropriate
user-facing message, so the chat page never needs to handle exceptions.
"""

from __future__ import annotations

from db.repository.session_factory import get_session
from db.repository.conversation_repository import ConversationRepository
from app.state.session_state import (
    ChatMessage,
    get_conversation_manager,
    get_db_url,
    get_schema,
    get_current_conversation_id,
    set_current_conversation_id,
)
from core.execution.query_executor import execute_query
from core.insights.insight_generator import generate_insight
from core.memory.conversation_manager import ConversationTurn
from core.nl2sql.sql_generator import generate_sql
from core.nl2sql.sql_validator import validate_sql
from core.schema.schema_context_builder import build_schema_context
from exceptions.domain_exceptions import (
    AIDataAnalystError,
    LLMAPIError,
    LLMTimeoutError,
    QueryExecutionError,
    QueryTimeoutError,
    SQLGenerationError,
)
from logging_setup.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(user_question: str, conversation_history: list[dict]) -> ChatMessage:
    """
    Runs one full question through the pipeline and returns a
    ChatMessage ready to be appended to the chat history and rendered.

    Args:
        user_question: the natural-language question from the user.
        conversation_history: prior chat turns for follow-up context.

    Returns:
        A ChatMessage with role="assistant" containing the result,
        or an error message if any step failed. Never raises.
    """
    db_url = get_db_url()

    # ── Step 1: Schema context ─────────────────────────────────────────────
    try:
        schema = get_schema(db_url)
        schema_context = build_schema_context(schema, user_question)
    except AIDataAnalystError as exc:
        return _error_message(str(exc), exc.user_message)
    except Exception as exc:
        return _error_message(str(exc), "Could not load the database schema.")

    # ── Step 2: SQL generation ─────────────────────────────────────────────
    try:
        gen_result = generate_sql(
            user_question=user_question,
            schema_context=schema_context,
            conversation_history=conversation_history,
        )
    except LLMTimeoutError as exc:
        _record_failure(user_question, exc.user_message)
        return _error_message(str(exc), exc.user_message)
    except LLMAPIError as exc:
        _record_failure(user_question, exc.user_message)
        return _error_message(str(exc), exc.user_message)
    except SQLGenerationError as exc:
        _record_failure(user_question, exc.user_message)
        return _error_message(str(exc), exc.user_message)
    except AIDataAnalystError as exc:
        _record_failure(user_question, exc.user_message)
        return _error_message(str(exc), exc.user_message)
    except Exception as exc:
        _record_failure(user_question, str(exc))
        return _error_message(str(exc), "The AI failed to generate a query. Please try again.")

    # Clarification requested — no SQL was generated.
    if gen_result.sql is None:
        clarification = gen_result.clarification_request or "Could you rephrase your question?"
        _record_failure(user_question, "Clarification needed")
        return ChatMessage(
            role="assistant",
            content=clarification,
        )

    # ── Step 3: SQL validation ─────────────────────────────────────────────
    dialect = schema.dialect if schema else "sqlite"
    validation = validate_sql(gen_result.sql, schema, dialect=dialect)

    if not validation.is_valid:
        _record_failure(user_question, validation.error_message or "Validation failed")
        return ChatMessage(
            role="assistant",
            content=(
                "I generated a query but it didn't pass the safety check. "
                "Please try rephrasing your question."
            ),
            sql=gen_result.sql,
            validation_error=validation.error_message,
        )

    cleaned_sql = validation.cleaned_sql or gen_result.sql

    # ── Step 4: SQL execution ──────────────────────────────────────────────
    try:
        query_result = execute_query(cleaned_sql, dialect=dialect)
    except QueryTimeoutError as exc:
        _record_failure(user_question, exc.user_message)
        return ChatMessage(
            role="assistant",
            content=exc.user_message,
            sql=cleaned_sql,
            error=str(exc),
        )
    except QueryExecutionError as exc:
        _record_failure(user_question, exc.user_message)
        return ChatMessage(
            role="assistant",
            content=exc.user_message,
            sql=cleaned_sql,
            error=str(exc),
        )
    except Exception as exc:
        _record_failure(user_question, str(exc))
        return ChatMessage(
            role="assistant",
            content="The query failed to execute. Please try again.",
            sql=cleaned_sql,
            error=str(exc),
        )

    # ── Step 5: Assemble result message ────────────────────────────────────
    row_count = query_result.row_count

    if row_count == 0:
        response_text = "The query ran successfully but returned no results."
    else:
        response_text = (
            f"Found **{row_count:,} row{'s' if row_count != 1 else ''}**."
        )

    # ── Step 6: Business insights (optional, non-fatal) ────────────────────
    # Generating insights is a "nice to have" narrative layer on top of
    # results that are already valid and already shown to the user.
    # A failure here must never prevent the user from seeing their
    # query results — generate_insight() itself never raises (see its
    # own docstring), but this is wrapped defensively anyway since a
    # pipeline step that silently depends on another module never
    # raising is a fragile assumption to build on unguarded.
    #
    # This runs BEFORE _record_success so the conversation turn stored
    # for memory/context purposes can include the insight summary in
    # the same record — requirement 1 of conversation memory is to
    # store the insight alongside the question/SQL/result, and doing
    # that in one write avoids a second mutation of the same turn.
    insight = None
    try:
        insight = generate_insight(
            user_question=user_question,
            generated_sql=cleaned_sql,
            result=query_result,
        )
    except Exception as exc:
        logger.warning("insight_step_failed_in_pipeline", error=str(exc)[:200])
        insight = None

    _record_success(
        question=user_question,
        sql=cleaned_sql,
        row_count=row_count,
        execution_time_ms=query_result.execution_time_ms,
        insight_summary=insight.summary if insight and not insight.is_empty else None,
    )
    
    session = get_session()
    repo = ConversationRepository(session)

    conversation_id = get_current_conversation_id()

    if conversation_id is None:
        conversation = repo.create_session(
            title=user_question[:50],
            database_url=db_url,
            llm_provider="groq",
            db_dialect=schema.dialect,
        )

        conversation_id = conversation.id
        set_current_conversation_id(conversation_id)

    

    repo.add_turn(
        conversation_id=conversation_id,
        user_question=user_question,
        assistant_response=response_text,
        generated_sql=cleaned_sql,
        validation_result="VALID",
        execution_status="SUCCESS",
        row_count=row_count,
        insight_text=insight.summary if insight and not insight.is_empty else None,
        chart_metadata=None,
    )

    

    session.close()

    
    return ChatMessage(
        role="assistant",
        content=response_text,
        sql=cleaned_sql,
        query_result=query_result,
        insight=insight,
    )


def _error_message(technical: str, user_facing: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=user_facing,
        error=technical,
    )


def _record_success(
    question: str,
    sql: str,
    row_count: int,
    execution_time_ms: float,
    insight_summary: str | None,
) -> None:

    # Existing in-memory conversation
    get_conversation_manager().add_turn(
        ConversationTurn(
            question=question,
            sql=sql,
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            insight_summary=insight_summary,
            success=True,
        )
    )

    # Temporary: verify repository works
    session = get_session()
    repo = ConversationRepository(session)

    print("Repository connected successfully")

    session.close()

def _record_failure(question: str, reason: str) -> None:
    get_conversation_manager().add_turn(ConversationTurn(
        question=question,
        sql=None,
        row_count=None,
        execution_time_ms=None,
        insight_summary=None,
        success=False,
        error_message=reason,
    ))
