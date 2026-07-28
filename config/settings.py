"""
config/settings.py
===================

Central, typed configuration for the entire application, built on
pydantic-settings.

WHY THIS EXISTS
----------------
Every other module in this project (db, core, app) reads configuration
through this module ONLY. No module should call `os.getenv()` directly.
This gives us:

  1. A single source of truth — one place to see every config value the
     app depends on, instead of `os.getenv()` calls scattered across
     30 files.
  2. Validation at startup, not at runtime. If GROQ_API_KEY is required
     and missing, we want the app to fail immediately on boot with a
     clear error — not three steps into a user's conversation when the
     LLM call finally happens.
  3. Type safety. `MAX_QUERY_ROWS` arrives as a string from the
     environment; pydantic coerces and validates it as an int once,
     here, instead of every call site doing its own int() and hoping.
  4. Easy environment swapping (dev/staging/prod) by pointing at
     different .env files without code changes.

HOW IT'S USED
-------------
    from config.settings import get_settings

    settings = get_settings()
    settings.database.url
    settings.llm.default_provider

`get_settings()` is cached (see bottom of file) so the .env file is
parsed once per process, not on every call.

NOTE: This file defines STRUCTURE only. Field validators and defaults
are sketched but the actual values come from the environment — see
.env.example for the variables this expects.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers. Adding a new provider means adding a
    value here AND a corresponding class in core/llm/providers/."""

    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


class DatabaseDialect(str, Enum):
    """Supported target database dialects for the USER'S connected
    database (not the app's own internal storage)."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class AppSettings(BaseSettings):
    """
    Top-level settings object. Pydantic-settings automatically reads
    matching environment variables (case-insensitive) and values from
    a .env file at the project root.

    Field naming convention: SCREAMING_SNAKE_CASE in .env maps to
    lower_snake_case attributes here via pydantic-settings defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = "AI Data Analyst"
    app_env: str = Field(default="development")  # development | staging | production
    debug: bool = Field(default=False)

    # --- LLM configuration ---
    # Groq is the only provider with a working implementation in V1 —
    # see core/llm/providers/groq_provider.py. The other three enum
    # values (claude/openai/gemini) still exist on LLMProvider and
    # still have stub classes registered for structural completeness
    # (so the factory's dispatch table doesn't need to change shape
    # later), but selecting them currently raises ConfigurationError
    # at get_llm_provider() time — see llm_client_factory.py.
    default_llm_provider: LLMProvider = Field(default=LLMProvider.GROQ)
    llm_request_timeout_seconds: int = Field(default=30)
    llm_max_retries: int = Field(default=2)

    anthropic_api_key: Optional[SecretStr] = Field(default=None)
    openai_api_key: Optional[SecretStr] = Field(default=None)
    google_api_key: Optional[SecretStr] = Field(default=None)
    groq_api_key: Optional[SecretStr] = Field(default=None)

    # --- Groq-specific tunables ---
    # These override config/llm_config.yaml's providers.groq.* defaults
    # when set. Per requirement #4 (configurable model/temperature/
    # max_tokens/timeout), these need to be settable via .env without
    # editing YAML — useful for a quick model swap or temperature
    # experiment without touching a tracked config file. None means
    # "use the YAML default"; an explicit env value here always wins.
    # See core/llm/llm_client_factory.py for the merge logic.
    groq_model: Optional[str] = Field(default=None)
    groq_temperature: Optional[float] = Field(default=None)
    groq_max_tokens: Optional[int] = Field(default=None)
    groq_timeout_seconds: Optional[float] = Field(default=None)

    # --- Database (the user's connected, queryable database) ---
    db_dialect: DatabaseDialect = Field(default=DatabaseDialect.SQLITE)
    database_url: str = Field(default="sqlite:///./data/chinook.db")
    db_connect_timeout_seconds: int = Field(default=10)
    db_query_timeout_seconds: int = Field(default=15)

    # SQLite-specific: whether connecting to a non-existent database
    # file should fail (default) or silently create a new empty file.
    # Default is False because this app's purpose is to query EXISTING
    # data — silently creating a new, empty database when a user
    # mistyped a path would hide what is almost certainly a user error
    # behind a confusing "0 tables found" experience three steps later,
    # rather than a clear error at connection time.
    sqlite_allow_create_if_missing: bool = Field(default=False)

    # Connection pool sizing for non-SQLite dialects (Postgres/MySQL,
    # Phase 2). SQLite uses SQLAlchemy's StaticPool/NullPool instead —
    # see db/connectors/sqlite_connector.py — so this is unused for V1
    # but defined now so PostgresConnector/MySQLConnector don't need a
    # settings.py change when Phase 2 lands.
    db_pool_size: int = Field(default=5)
    db_pool_max_overflow: int = Field(default=10)

    # --- Internal app storage (conversation logs, query history, etc.) ---
    internal_db_url: str = Field(default="sqlite:///./data/app_internal.db")

    # --- Safety / SQL execution guardrails ---
    max_query_rows: int = Field(default=1000)
    enforce_read_only_role: bool = Field(default=True)
    allowed_sql_statement_types: tuple[str, ...] = Field(default=("SELECT", "WITH"))

    # --- Schema RAG (for large databases) ---
    schema_rag_enabled: bool = Field(default=False)
    schema_rag_top_k_tables: int = Field(default=8)
    vector_store_path: str = Field(default="./data/schema_index")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json | console
    log_file_path: Optional[str] = Field(default="./logs/app.log")

    # --- Conversation ---
    max_conversation_turns_in_context: int = Field(default=10)


@lru_cache
def get_settings() -> AppSettings:
    """
    Returns a cached singleton AppSettings instance.

    Cached via lru_cache so the .env file is parsed exactly once per
    process. Tests that need different settings should use
    `get_settings.cache_clear()` plus environment monkeypatching, or
    construct `AppSettings(**overrides)` directly rather than relying
    on this singleton.
    """
    return AppSettings()
