"""
core/nl2sql/security_rules_loader.py
========================================

Loads and caches config/security_rules.yaml into a typed
SecurityRules dataclass used exclusively by sql_validator.py.

WHY A SEPARATE LOADER MODULE INSTEAD OF READING YAML IN THE VALIDATOR
------------------------------------------------------------------------
Same separation-of-concerns reasoning used in
core/llm/llm_config_loader.py: the validator file should express
validation logic, not file-parsing/caching logic. Keeping them
separate also means the validator can be tested with a custom
SecurityRules instance (no filesystem access required), while this
loader is tested independently to confirm it reads the YAML correctly.

WHY YAML-DRIVEN RATHER THAN HARDCODED IN PYTHON
-------------------------------------------------
Security policy (which keywords are forbidden, what the row cap is)
belongs in a file that is:
  - Human-readable without Python knowledge — a DBA can audit it
  - Cleanly diffable in version control — "this PR added MERGE to
    forbidden_keywords" is a one-line, reviewable change
  - Reloadable per environment without code changes

See config/security_rules.yaml's own header comment for the full
rationale. This loader reads that file once and caches it; the
validator never reads YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from exceptions.domain_exceptions import ConfigurationError
from logging_setup.logger import get_logger

logger = get_logger(__name__)

_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "security_rules.yaml"
_cache: dict[str, Any] | None = None


@dataclass(frozen=True)
class SchemaAccessConfig:
    """Access-control configuration for tables and columns."""

    mode: str  # "allow_all" | "explicit_allow_list" | "deny_list"
    allowed_tables: frozenset[str]
    denied_tables: frozenset[str]
    denied_columns: frozenset[str]  # "table.column" format


@dataclass(frozen=True)
class SecurityRules:
    """
    Complete, typed security policy loaded from security_rules.yaml.
    Passed into validate_sql() so the validator never reads config
    files itself — makes the validator fully testable without a
    filesystem.
    """

    allowed_statement_types: frozenset[str]
    forbidden_keywords: frozenset[str]
    reject_sql_comments: bool
    reject_multiple_statements: bool
    max_rows_returned: int
    schema_access: SchemaAccessConfig


def load_security_rules(rules_path: Path | None = None) -> SecurityRules:
    """
    Loads security_rules.yaml and returns a typed SecurityRules object.
    Results are cached for the process lifetime — the file changes only
    at deployment, never at runtime.

    Args:
        rules_path: overrides the default path; used in tests that
            need to load a custom rules file without touching the real
            config/security_rules.yaml.

    Raises:
        ConfigurationError: the file is missing or malformed.
    """
    global _cache

    path = rules_path or _RULES_PATH

    # Only cache when using the default path — tests using a custom path
    # should always get a fresh load.
    use_cache = rules_path is None
    if use_cache and _cache is not None:
        return _cache  # type: ignore[return-value]

    if not path.is_file():
        raise ConfigurationError(
            message=f"Security rules file not found at {path}",
            user_message="The application is missing required security configuration.",
        )

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            message=f"Failed to parse security rules YAML at {path}: {exc}",
            user_message="The application's security configuration is invalid.",
        ) from exc

    try:
        sql_safety = raw["sql_safety"]
        schema_access_raw = raw.get("schema_access", {})

        rules = SecurityRules(
            allowed_statement_types=frozenset(
                s.upper() for s in sql_safety["allowed_statement_types"]
            ),
            forbidden_keywords=frozenset(
                k.upper() for k in sql_safety["forbidden_keywords"]
            ),
            reject_sql_comments=bool(sql_safety.get("reject_sql_comments", True)),
            reject_multiple_statements=bool(sql_safety.get("reject_multiple_statements", True)),
            max_rows_returned=int(sql_safety.get("max_rows_returned", 1000)),
            schema_access=SchemaAccessConfig(
                mode=schema_access_raw.get("mode", "allow_all"),
                allowed_tables=frozenset(schema_access_raw.get("allowed_tables") or []),
                denied_tables=frozenset(schema_access_raw.get("denied_tables") or []),
                denied_columns=frozenset(schema_access_raw.get("denied_columns") or []),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            message=f"security_rules.yaml has an unexpected structure: {exc}",
            user_message="The application's security configuration is invalid.",
        ) from exc

    logger.debug(
        "security_rules_loaded",
        forbidden_keyword_count=len(rules.forbidden_keywords),
        max_rows=rules.max_rows_returned,
    )

    if use_cache:
        _cache = rules  # type: ignore[assignment]

    return rules


def clear_cache() -> None:
    """Clears the cached rules. Used in tests."""
    global _cache
    _cache = None
