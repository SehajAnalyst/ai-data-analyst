"""
exceptions/domain_exceptions.py
=================================

Domain-specific exceptions, grouped by the subsystem that raises them.
All inherit from AIDataAnalystError (see base.py).

WHY GROUPED THIS WAY (and not one file per exception)
-------------------------------------------------------
One class per file would be excessive for what are essentially typed
error tags. Grouping by subsystem here means: when you're working in
core/nl2sql/, you import from the "SQL generation & validation" section
below and immediately see every error that subsystem can raise, in one
place — useful both for writing `except` clauses and for onboarding.

USAGE PATTERN
-------------
Each subsystem module (sql_validator.py, query_executor.py, etc.)
raises ONLY exceptions defined here — never raw `ValueError` or
`Exception` — so that calling code can handle failures precisely:

    try:
        validated_sql = validator.validate(raw_sql)
    except SQLSafetyViolation as e:
        # specifically a safety rejection — log as a security event,
        # show user_message, optionally trigger a regeneration retry
    except SQLValidationError as e:
        # syntactic/schema validation failure — different handling
"""

from __future__ import annotations

from exceptions.base import AIDataAnalystError


# --- Configuration ---------------------------------------------------

class ConfigurationError(AIDataAnalystError):
    """Raised when required configuration is missing or invalid at
    startup (e.g. selected LLM provider has no API key configured)."""


# --- LLM layer (core/llm/) -------------------------------------------

class LLMProviderError(AIDataAnalystError):
    """Base class for all LLM-provider-related failures."""


class LLMAPIError(LLMProviderError):
    """The provider's API call failed (network, auth, rate limit, 5xx)."""


class LLMTimeoutError(LLMProviderError):
    """The LLM call exceeded the configured timeout."""


class LLMResponseParsingError(LLMProviderError):
    """The LLM's response could not be parsed into the expected
    structure (e.g. expected JSON, got malformed output)."""


# --- Schema introspection (core/schema/) ------------------------------

class SchemaIntrospectionError(AIDataAnalystError):
    """Failed to read schema metadata from the connected database.

    Raised for genuine failures: corrupt file, permission denied,
    connection dropped mid-reflection. NOT raised for a database that
    connects fine but simply has zero tables — see EmptyDatabaseError
    for that case, which is a normal state, not a failure.
    """


class EmptyDatabaseError(AIDataAnalystError):
    """The connected database has zero tables.

    Deliberately a DISTINCT exception type from SchemaIntrospectionError,
    not a subclass and not reused — introspection here SUCCEEDED; the
    database is just empty. Conflating this with a genuine introspection
    failure would make callers unable to tell "something is broken" from
    "this is a valid but empty database," which need different UI
    treatment (a clear empty-state message vs. an error banner).

    Whether this is raised at all is the caller's choice — see
    introspect_schema()'s `raise_on_empty` parameter. Some callers (the
    Schema Explorer page) want to display an empty state without an
    exception; others may want to fail fast.
    """


class SchemaNotFoundError(AIDataAnalystError):
    """Referenced table/column does not exist in the introspected
    schema. Distinct from SQLSafetyViolation: this is about
    correctness, not safety."""


# --- SQL generation & validation (core/nl2sql/) -----------------------

class SQLGenerationError(AIDataAnalystError):
    """The LLM failed to produce SQL after all retry attempts."""


class SQLValidationError(AIDataAnalystError):
    """Generated SQL failed syntactic or schema validation (e.g.
    references a column that doesn't exist). Recoverable via
    regeneration with the error fed back to the LLM."""


class SQLSafetyViolation(AIDataAnalystError):
    """Generated SQL violated a hard safety rule (non-SELECT statement,
    forbidden keyword, multiple statements, etc).

    IMPORTANT: this should always also be logged as a security event
    (see logging_setup), distinct from ordinary application errors,
    since repeated occurrences may indicate prompt injection attempts
    rather than benign model mistakes.
    """


# --- Query execution (core/execution/) ---------------------------------

class QueryExecutionError(AIDataAnalystError):
    """The validated, approved SQL failed during execution against the
    database (e.g. transient connection failure)."""


class QueryTimeoutError(QueryExecutionError):
    """Query exceeded the configured execution timeout and was killed."""


class RowLimitExceededError(QueryExecutionError):
    """Result set exceeded the configured maximum row limit even after
    LIMIT injection — should not normally occur; indicates a validator
    or injection bug if it does."""


# --- Database connection (db/) ------------------------------------------

class DatabaseConnectionError(AIDataAnalystError):
    """Could not establish or maintain a connection to the target
    database."""


class DatabaseFileNotFoundError(DatabaseConnectionError):
    """SQLite database file does not exist at the configured path and
    sqlite_allow_create_if_missing is False.

    Kept as a distinct subclass of DatabaseConnectionError (rather
    than reusing it directly) because the two failure modes need
    different user-facing messages: "file not found, check the path"
    is an actionable, specific fix; a generic "could not connect" is
    not. Callers that only care about connection failure broadly can
    still catch DatabaseConnectionError and handle both uniformly.
    """


class UnsupportedDialectError(AIDataAnalystError):
    """Requested database dialect has no registered connector."""


# --- ML plugins (ml_plugins/) --------------------------------------------

class MLPluginError(AIDataAnalystError):
    """Base class for ML plugin failures."""


class PluginNotFoundError(MLPluginError):
    """Requested ML capability has no registered plugin."""


class PluginValidationError(MLPluginError):
    """Input data does not satisfy the plugin's requirements (e.g.
    missing required columns, insufficient rows for the model type)."""


# --- Orchestration (core/orchestration/) -----------------------------------

class IntentClassificationError(AIDataAnalystError):
    """Could not confidently classify user intent (new query vs.
    follow-up vs. ML request vs. out-of-scope)."""


class ConversationStateError(AIDataAnalystError):
    """Conversation/session state is missing or corrupted."""
