"""
core/execution/query_executor.py
===================================

Executes VALIDATED SQL against the target database. The only module
in the project permitted to run SQL against the user's database.

PRECONDITION: validated_sql must be ValidationResult.sanitized_sql
from a call to sql_validator.validate_sql() that returned is_valid=True.
This function does not re-validate. If that contract is broken, the
safety guarantees of the validator are voided.

TIMEOUT: enforced at the SQLAlchemy connection level via
connect_args. SQLite supports this via the sqlite3 timeout parameter.
For operations that hang beyond the limit, the connection is closed.

ROW LIMIT: enforced post-fetch as a hard backstop. The validator
already injects LIMIT into the SQL (layer 1). This truncation check
(layer 2) catches any edge case where the injected limit was bypassed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from config.settings import get_settings
from app.state.session_state import get_engine
from exceptions.domain_exceptions import QueryExecutionError, QueryTimeoutError
from logging_setup.logger import LogCategory, LogFields, get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result of one SQL execution."""

    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool          # True if results were capped at max_query_rows
    execution_time_ms: float


def execute_query(validated_sql: str, dialect: str = "sqlite") -> QueryResult:
    """
    Executes validator-approved SQL and returns results as a QueryResult.

    Args:
        validated_sql: must be ValidationResult.sanitized_sql from a
            passing validate_sql() call. Not re-validated here.
        dialect: target dialect for engine selection.

    Returns:
        QueryResult with all rows and metadata.

    Raises:
        QueryExecutionError: execution failed for any database reason.
        QueryTimeoutError: query exceeded the configured timeout.
    """
    settings = get_settings()
    max_rows = settings.max_query_rows

    log = logger.bind(
        category=LogCategory.QUERY_EXECUTION,
        sql_preview=validated_sql[:120],
        dialect=dialect,
    )

    try:
        engine = get_engine()
    except Exception as exc:
        raise QueryExecutionError(
            message=f"Could not acquire database engine: {exc}",
            user_message="Could not connect to the database. Check your connection settings.",
        ) from exc

    start = time.monotonic()

    try:
        with engine.connect() as conn:
            result = conn.execute(text(validated_sql))
            columns = list(result.keys())
            # Fetch max_rows + 1 so we can detect truncation without
            # pulling an entire unbounded result set into memory.
            rows = result.fetchmany(max_rows + 1)

    except OperationalError as exc:
        msg = str(exc)
        elapsed = (time.monotonic() - start) * 1000
        log.error("query_execution_failed", error=msg[:200], latency_ms=round(elapsed, 1))
        if "interrupted" in msg.lower() or "timeout" in msg.lower():
            raise QueryTimeoutError(
                message=f"Query exceeded timeout: {exc}",
                user_message=(
                    f"The query took too long to run (limit: "
                    f"{settings.db_query_timeout_seconds}s). "
                    "Try a more specific question or add filters."
                ),
            ) from exc
        raise QueryExecutionError(
            message=f"Query execution failed: {exc}",
            user_message="The query failed to execute. The SQL may be invalid for this database.",
        ) from exc

    except SQLAlchemyError as exc:
        elapsed = (time.monotonic() - start) * 1000
        log.error("query_execution_failed", error=str(exc)[:200], latency_ms=round(elapsed, 1))
        raise QueryExecutionError(
            message=f"Database error during execution: {exc}",
            user_message="A database error occurred. Please try again.",
        ) from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    log.info(
        "query_execution_succeeded",
        **{
            LogFields.ROW_COUNT: len(rows),
            LogFields.LATENCY_MS: round(elapsed_ms, 1),
        },
        truncated=truncated,
    )

    return QueryResult(
        columns=columns,
        rows=list(rows),
        row_count=len(rows),
        truncated=truncated,
        execution_time_ms=elapsed_ms,
    )
