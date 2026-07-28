"""
logging_setup/logger.py
=========================

Centralized logging configuration for the entire application.

WHY CENTRALIZED, AND WHY STRUCTURED (JSON) LOGGING
-----------------------------------------------------
This app has a specific debugging problem ordinary apps don't: most of
its "logic" happens inside LLM calls, which are non-deterministic and
opaque. When something goes wrong — wrong SQL, a safety rejection, a
confusing answer — you need to reconstruct: what schema context was
sent, what prompt was sent, what the LLM returned, what the validator
did with it, and what executed. A single unstructured log line per
event won't support that kind of reconstruction. Structured (JSON)
logging with consistent fields lets you grep/filter by session_id or
trace_id and pull the entire chain for one user turn.

For the database connection module specifically: connection failures
are exactly the kind of thing that gets debugged days later from logs
("why did this user's session fail at 3am") — so every connection
attempt, success, and failure is logged with enough structured detail
(dialect, file path for SQLite, read_only flag, latency) to answer
that without reproducing the bug live.

WHY NOT print() OR DEFAULT logging.basicConfig() SCATTERED PER-MODULE
-------------------------------------------------------------------------
If every module configures its own logger ad hoc, you get inconsistent
formats, duplicate handlers, and no way to globally change log level or
destination (e.g. "send WARNING+ to a file in prod") without touching
every file. This module is the ONLY place logging is configured;
every other module just does:

    from logging_setup.logger import get_logger
    logger = get_logger(__name__)

CONFIGURATION
-------------
`configure_logging()` must be called exactly once, at process startup
(app/main.py's bootstrap()). It reads settings.log_level/log_format/
log_file_path and wires structlog + the stdlib logging module
accordingly. Calling get_logger() before configure_logging() has run
still works (it auto-configures on first use), but explicit startup
configuration is still preferred so configuration happens
deterministically before any module-level logger calls fire.

LOG CATEGORIES (used as a `category` field on every log call)
-----------------------------------------------------------------
  - "llm_call"        : every LLM request/response (latency, tokens, provider)
  - "sql_generation"   : generated SQL + the question that produced it
  - "sql_validation"   : validation outcome (pass/reject + reason)
  - "security"         : SQL safety violations — these should be
                           monitorable/alertable independently of
                           ordinary errors (see note in domain_exceptions.py)
  - "query_execution"  : execution timing, row counts, timeouts
  - "db_connection"    : connection attempts, successes, failures —
                           used by db/connectors/
  - "user_action"      : UI-level user actions, for usage analytics later

CORRELATION
-----------
Every log line within one user turn should carry the same
`session_id` and `trace_id` so logs can be reconstructed end-to-end.
trace_id is generated once per user question, not once per session,
so multi-turn conversations can still be inspected turn-by-turn.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_configured = False


def configure_logging() -> None:
    """
    Configures structlog + stdlib logging for the whole process.
    Idempotent — safe to call more than once (e.g. across Streamlit
    reruns); only the first call has an effect.

    Reads configuration from config.settings.get_settings():
        log_level     -> stdlib logging level threshold
        log_format     -> "json" (production/aggregated logs) or
                           "console" (human-readable, local dev)
        log_file_path  -> if set, logs also go to this file (in
                           addition to stderr); directory is created
                           if missing

    Must be called once at process startup before any meaningful
    logging happens — see app/main.py's bootstrap().
    """
    global _configured
    if _configured:
        return

    # Imported here, not at module top level, to avoid a circular
    # import: config.settings has no dependency on logging_setup, but
    # keeping the import local to this function makes that boundary
    # explicit and avoids accidental future coupling.
    from config.settings import get_settings

    settings = get_settings()

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if settings.log_file_path:
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(message)s",  # structlog renders the actual message shape
        handlers=handlers,
        force=True,  # override any prior basicConfig (e.g. from a library)
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Returns a configured logger for the given module name.

    Convention: always call with `__name__` so log lines are traceable
    to their source module:

        logger = get_logger(__name__)
        logger.info(
            "db_connection_established",
            category=LogCategory.DB_CONNECTION,
            dialect="sqlite",
            latency_ms=4.2,
        )

    Note the event-name-first style (first positional arg is a short,
    snake_case event name, not a sentence) — this is structlog
    convention and what makes JSON log output greppable/filterable by
    event type, not just free-text message matching.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def bind_session_context(session_id: str, trace_id: str) -> None:
    """
    Binds session_id/trace_id into structlog's contextvars so every
    subsequent log call on this async/thread context automatically
    includes them, without each call site passing them explicitly.

    WHY THIS EXISTS AS ITS OWN FUNCTION: session/trace binding needs
    to happen exactly once per turn, at the top of
    ConversationManager's handling of a new question — not
    re-derived ad hoc in every downstream module. Centralizing it
    here prevents drift (e.g. one module using "session" as the key
    and another using "session_id").

    Call clear_session_context() at the end of the turn (or start of
    the next one) to avoid context leaking across unrelated turns
    within the same process.
    """
    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)


def clear_session_context() -> None:
    """Clears bound session/trace context. Call at the end of a turn,
    or Streamlit-session-start, to avoid leaking context across turns."""
    structlog.contextvars.clear_contextvars()


# Standard field names — used as a contract so every module logs the
# same field names rather than inventing their own per call site.
class LogFields:
    SESSION_ID = "session_id"
    TRACE_ID = "trace_id"
    CATEGORY = "category"
    LLM_PROVIDER = "llm_provider"
    LLM_MODEL = "llm_model"
    LATENCY_MS = "latency_ms"
    TOKEN_COUNT_INPUT = "token_count_input"
    TOKEN_COUNT_OUTPUT = "token_count_output"
    SQL_TEXT = "sql_text"
    VALIDATION_RESULT = "validation_result"
    REJECTION_REASON = "rejection_reason"
    ROW_COUNT = "row_count"
    DB_DIALECT = "dialect"
    DB_PATH = "db_path"
    READ_ONLY = "read_only"
    TABLE_NAME = "table_name"
    TABLE_COUNT = "table_count"
    COLUMN_COUNT = "column_count"
    RELATIONSHIP_COUNT = "relationship_count"


class LogCategory:
    LLM_CALL = "llm_call"
    SQL_GENERATION = "sql_generation"
    SQL_VALIDATION = "sql_validation"
    SECURITY = "security"
    QUERY_EXECUTION = "query_execution"
    DB_CONNECTION = "db_connection"
    SCHEMA_DISCOVERY = "schema_discovery"
    USER_ACTION = "user_action"
