"""
core/nl2sql/sql_validator.py
================================

Deterministic, non-LLM SQL validation and security layer.

This is the most safety-critical file in the project. Read every
docstring here before modifying anything, and read the GROUNDING NOTES
below before assuming how sqlglot behaves.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITION IN THE PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  sql_generator.py → [THIS FILE] → query_executor.py

sql_generator.py produces SQL. This file decides whether that SQL is
safe and correct. query_executor.py runs it. The executor never
receives SQL that hasn't passed this validator — that boundary is
what makes this a real checkpoint, not a formality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY AST-BASED, NOT REGEX-BASED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Regex-based "does this string contain DROP" is routinely defeated:

  SELECT * FROM employees --
  DROP TABLE employees

The regex sees "DROP" and rejects. But consider:

  SELECT * FROM drop_log

The regex false-positives on the column name. Or:

  SELECT * FROM employees WHERE name = 'DROP TABLE employees'

The regex rejects a valid, safe query because "DROP" is inside a
string literal. AST parsing removes both problems: we inspect the
*structure* of the SQL, not its text. A DROP node in the AST is a
real DROP statement. The word "drop" inside an identifier or literal
does not produce a Drop AST node.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING NOTES (verified against sqlglot 30.x before writing this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. COMMENT STRIPPING: sqlglot strips SQL comments before parsing.
   "SELECT * FROM employees -- DROP TABLE employees" parses cleanly
   as a Select with no evidence the comment existed. The comment
   check MUST happen on the raw string BEFORE any sqlglot call.

2. CTE ALIASES: a CTE named "dept_stats" appears as an exp.Table
   node just like a real table. Without explicit exclusion, every
   CTE query would false-positive on schema validation ("dept_stats
   not in schema"). All CTE alias names are extracted from exp.CTE
   nodes and excluded from the real-table check.

3. TABLE ALIASES: column references through aliases (e.name when
   'employees AS e') have col.table = 'e', not 'employees'. Column
   validation builds an alias→real_table map first and resolves
   through it before checking schema.

4. MULTIPLE STATEMENTS: sqlglot.parse() returns a list. A stacked
   injection "SELECT 1; DROP TABLE t" returns two items. Checking
   len(statements) > 1 catches this at the parse level, before any
   per-statement analysis.

5. LIMIT INJECTION: ast.limit(N) produces a new AST node with LIMIT N
   appended. Calling .sql(dialect=dialect) on the modified AST gives
   syntactically correct output. Existing LIMIT values are checked
   against max_rows; if the existing LIMIT exceeds the cap, it is
   replaced, not trusted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECK ORDER (first failure short-circuits)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Empty / whitespace-only input
  2. Comment presence (raw string scan — see grounding note #1)
  3. Multiple statements (sqlglot.parse list length)
  4. SQL injection pattern scan (UNION-based, stacked comment tricks)
  5. Parse succeeds (sqlglot; parse failure = syntax_error category)
  6. Statement type is SELECT or WITH...SELECT (not DROP/DELETE/etc.)
  7. Forbidden keyword nodes in AST (catches obfuscated variants)
  8. Schema access policy: table deny-list / allow-list
  9. Hallucinated tables (real tables not in schema)
 10. Hallucinated columns (qualified col refs not in schema)
 11. LIMIT enforcement: inject if missing, cap if over max

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS MODULE DOES NOT DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

It does not execute anything. It does not replace the database-level
read-only role enforced in db/connectors/sqlite_connector.py — that
is layer #2 and is required independently. This is layer #1. Both
must be present; neither substitutes for the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from core.nl2sql.security_rules_loader import SecurityRules, load_security_rules
from core.schema.schema_introspector import DatabaseSchema
from logging_setup.logger import LogCategory, LogFields, get_logger

logger = get_logger(__name__)

# ── AST node types that map to forbidden operations ──────────────────────────
_FORBIDDEN_NODE_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Drop,
    exp.Delete,
    exp.Insert,
    exp.Update,
    exp.Create,
    exp.Alter,         # ALTER TABLE / ALTER INDEX / ALTER SESSION / etc.
    exp.Grant,
    exp.Command,       # catches EXEC, EXECUTE, PRAGMA, ATTACH, DETACH
    exp.Transaction,   # BEGIN/COMMIT/ROLLBACK — not dangerous but out of scope
    exp.Merge,
)

# ── Raw-string injection patterns checked BEFORE parsing ─────────────────────
# These catch obfuscation tricks that might survive into the AST or that
# rely on features sqlglot normalises away (e.g. MySQL's /*!...*/ comments).
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"/\*!", re.IGNORECASE),
        "MySQL conditional comment syntax (/*!...*/) is not permitted.",
    ),
    (
        re.compile(r";\s*--", re.IGNORECASE),
        "Statement terminator followed by comment is a known injection pattern.",
    ),
    (
        re.compile(r"\bxp_\w+\b", re.IGNORECASE),
        "SQL Server extended stored procedure calls (xp_) are not permitted.",
    ),
    (
        re.compile(r"\bsys\.\w+\b", re.IGNORECASE),
        "System catalog references (sys.*) are not permitted.",
    ),
    (
        re.compile(r"\binformation_schema\b", re.IGNORECASE),
        "INFORMATION_SCHEMA references are not permitted.",
    ),
    (
        re.compile(r"\bsqlite_master\b|\bsqlite_schema\b", re.IGNORECASE),
        "Internal SQLite system table references are not permitted.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """
    Complete output of one validate_sql() call.

    Fields:
        is_valid: True only if every check passed. When False, all
            other substantive fields (cleaned_sql, detected_tables,
            detected_columns) are still populated where possible, so
            callers can log what was found even for rejected queries.
        original_sql: the raw SQL string as received — stored
            unmodified for logging and audit purposes regardless of
            whether validation passed or failed.
        cleaned_sql: the final, executor-ready SQL string. For valid
            queries this differs from original_sql only in that:
              - A LIMIT clause has been injected if absent.
              - An oversized LIMIT has been replaced with the cap.
            None when is_valid is False.
        error_message: human-readable rejection reason. Safe to show
            to the user in the UI AND to feed back into
            sql_generator.py's retry prompt. Does NOT include raw
            schema details (table list, column list) that would
            constitute an information disclosure if logged or
            displayed without care — specific enough for the LLM to
            self-correct, not so verbose as to leak internals.
        rejection_category: one of:
              "safety_violation"  — dangerous keyword/statement type
              "injection_pattern" — raw-string injection signal
              "schema_mismatch"   — hallucinated table or column
              "syntax_error"      — sqlglot could not parse the SQL
              "access_denied"     — table/column on the deny-list
            None when is_valid is True. Callers MUST log
            safety_violation and injection_pattern events separately
            from ordinary validation failures — repeated occurrences
            may indicate active prompt injection attempts.
        detected_tables: real (non-CTE-alias) table names the
            validator found in the AST. Populated for both valid and
            invalid queries (useful for logging rejected queries).
        detected_columns: qualified column references found in the
            AST, as "table.column" strings. Unqualified references
            (no table prefix) are listed as ".column". Populated for
            both valid and invalid queries.
    """

    is_valid: bool
    original_sql: str
    cleaned_sql: str | None
    error_message: str | None
    rejection_category: str | None
    detected_tables: list[str] = field(default_factory=list)
    detected_columns: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def validate_sql(
    raw_sql: str,
    schema: DatabaseSchema,
    dialect: str = "sqlite",
    rules: SecurityRules | None = None,
) -> ValidationResult:
    """
    Validates LLM-generated SQL through every safety and correctness
    check documented in the module header, in the order listed there.

    Args:
        raw_sql: SQL text from sql_generator.py. Treat as untrusted.
        schema: introspected DatabaseSchema from the Schema Discovery
            module. Used for table and column existence checks.
        dialect: target SQL dialect ("sqlite", "postgresql", "mysql").
            Controls sqlglot's parser and the SQL generated for
            LIMIT injection.
        rules: SecurityRules from security_rules_loader. If None,
            loads from config/security_rules.yaml automatically. Pass
            explicitly in tests to avoid filesystem access.

    Returns:
        ValidationResult. When is_valid is True, cleaned_sql is safe
        to pass directly to query_executor.py. When is_valid is False,
        error_message is safe to feed to sql_generator.py's retry
        prompt and to display in the UI.

    Does NOT raise exceptions for validation failures — all outcomes
    are expressed through ValidationResult. Raises only for genuine
    infrastructure failures (e.g. security_rules.yaml missing), which
    are the caller's responsibility to handle separately.
    """
    if rules is None:
        rules = load_security_rules()

    log = logger.bind(
        category=LogCategory.SQL_VALIDATION,
        dialect=dialect,
        sql_preview=raw_sql[:120] if raw_sql else "",
    )

    # ── Check 1: empty input ──────────────────────────────────────────────────
    if not raw_sql or not raw_sql.strip():
        log.warning("sql_validation_rejected", reason="empty_input")
        return _reject(
            raw_sql,
            error_message="SQL query cannot be empty.",
            category="syntax_error",
        )

    # ── Check 2: comment presence (raw string — grounding note #1) ───────────
    if rules.reject_sql_comments and _has_comments(raw_sql):
        log.error(
            "sql_validation_rejected",
            category=LogCategory.SECURITY,
            reason="sql_comments_present",
        )
        return _reject(
            raw_sql,
            error_message=(
                "SQL comments (-- or /* */) are not permitted. "
                "Comments can be used to disguise injected statements."
            ),
            category="safety_violation",
        )

    # ── Check 3 + 4: injection patterns (raw string) ─────────────────────────
    pattern_error = _check_injection_patterns(raw_sql)
    if pattern_error:
        log.error(
            "sql_validation_rejected",
            category=LogCategory.SECURITY,
            reason="injection_pattern",
        )
        return _reject(
            raw_sql,
            error_message=pattern_error,
            category="injection_pattern",
        )

    # ── Check 5: parse with sqlglot ───────────────────────────────────────────
    try:
        statements = sqlglot.parse(raw_sql, dialect=dialect, error_level=sqlglot.ErrorLevel.RAISE)
    except sqlglot.errors.ParseError as exc:
        log.warning("sql_validation_rejected", reason="parse_error", error=str(exc)[:200])
        return _reject(
            raw_sql,
            error_message=f"SQL could not be parsed: {exc}",
            category="syntax_error",
        )

    # ── Check 6: multiple statements ─────────────────────────────────────────
    if rules.reject_multiple_statements and len(statements) > 1:
        log.error(
            "sql_validation_rejected",
            category=LogCategory.SECURITY,
            reason="multiple_statements",
            statement_count=len(statements),
        )
        return _reject(
            raw_sql,
            error_message=(
                f"Multiple SQL statements detected ({len(statements)}). "
                "Only a single SELECT statement is permitted per query."
            ),
            category="safety_violation",
        )

    if not statements or statements[0] is None:
        return _reject(
            raw_sql,
            error_message="SQL parsed to an empty statement.",
            category="syntax_error",
        )

    ast = statements[0]

    # ── Check 7: statement type ───────────────────────────────────────────────
    statement_type = type(ast).__name__.upper()
    if statement_type not in rules.allowed_statement_types:
        log.error(
            "sql_validation_rejected",
            category=LogCategory.SECURITY,
            reason="forbidden_statement_type",
            statement_type=statement_type,
        )
        return _reject(
            raw_sql,
            error_message=(
                f"Statement type '{statement_type}' is not permitted. "
                "Only SELECT queries (including CTEs) are allowed."
            ),
            category="safety_violation",
        )

    # ── Check 8: forbidden AST node types ────────────────────────────────────
    forbidden_node = _find_forbidden_node(ast)
    if forbidden_node is not None:
        node_name = type(forbidden_node).__name__
        log.error(
            "sql_validation_rejected",
            category=LogCategory.SECURITY,
            reason="forbidden_ast_node",
            node_type=node_name,
        )
        return _reject(
            raw_sql,
            error_message=(
                f"Forbidden SQL operation detected: {node_name}. "
                "Only read-only SELECT queries are permitted."
            ),
            category="safety_violation",
        )

    # ── Extract structural facts from the AST ─────────────────────────────────
    cte_aliases = _extract_cte_aliases(ast)
    alias_map = _build_alias_map(ast)
    real_tables = _extract_real_tables(ast, cte_aliases)
    column_refs = _extract_column_refs(ast, alias_map)

    # ── Check 9: schema access policy ────────────────────────────────────────
    access_error = _check_access_policy(real_tables, rules)
    if access_error:
        log.warning("sql_validation_rejected", reason="access_policy", detail=access_error[:200])
        return _reject(
            raw_sql,
            error_message=access_error,
            category="access_denied",
            detected_tables=sorted(real_tables),
            detected_columns=_format_column_refs(column_refs),
        )

    # ── Check 10: hallucinated tables ─────────────────────────────────────────
    table_error = _check_tables_in_schema(real_tables, schema)
    if table_error:
        log.warning(
            "sql_validation_rejected",
            reason="hallucinated_table",
            **{LogFields.SQL_TEXT: raw_sql[:200]},
        )
        return _reject(
            raw_sql,
            error_message=table_error,
            category="schema_mismatch",
            detected_tables=sorted(real_tables),
            detected_columns=_format_column_refs(column_refs),
        )

    # ── Check 11: hallucinated columns ────────────────────────────────────────
    column_error = _check_columns_in_schema(column_refs, schema)
    if column_error:
        log.warning(
            "sql_validation_rejected",
            reason="hallucinated_column",
            **{LogFields.SQL_TEXT: raw_sql[:200]},
        )
        return _reject(
            raw_sql,
            error_message=column_error,
            category="schema_mismatch",
            detected_tables=sorted(real_tables),
            detected_columns=_format_column_refs(column_refs),
        )

    # ── Check 12: LIMIT enforcement ───────────────────────────────────────────
    ast, limit_injected = _enforce_limit(ast, rules.max_rows_returned, dialect)
    if limit_injected:
        log.debug("sql_validation_limit_applied", max_rows=rules.max_rows_returned)

    cleaned_sql = ast.sql(dialect=dialect)

    log.info(
        "sql_validation_passed",
        detected_table_count=len(real_tables),
        **{LogFields.VALIDATION_RESULT: "pass"},
    )

    return ValidationResult(
        is_valid=True,
        original_sql=raw_sql,
        cleaned_sql=cleaned_sql,
        error_message=None,
        rejection_category=None,
        detected_tables=sorted(real_tables),
        detected_columns=_format_column_refs(column_refs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal check functions — one function per distinct concern
# ─────────────────────────────────────────────────────────────────────────────


def _has_comments(sql: str) -> bool:
    """
    Checks raw SQL text for comment syntax BEFORE parsing.

    Must run on the raw string because sqlglot strips comments before
    building the AST (verified in grounding tests) — once parsed, there
    is no evidence comments were present.

    Checks both:
      -- line comments (anywhere after two dashes)
      /* block comments */ (any occurrence of the opening marker)

    Does NOT use regex here intentionally — a simple substring check is
    faster, has no edge cases around flag combinations, and comments
    can't be disguised in a way that fools a literal substring search
    (the parser itself would have to see the comment marker to treat it
    as a comment, so if the parser would see it, so does this check).
    """
    return "--" in sql or "/*" in sql


def _check_injection_patterns(sql: str) -> str | None:
    """
    Scans raw SQL text for known injection patterns that either
    survive sqlglot normalisation or rely on database-specific
    features sqlglot doesn't model (MySQL conditional comments,
    SQL Server xp_ procedures, etc.).

    Returns the first matching error message string, or None if
    no patterns matched. Called before parsing so that patterns
    intentionally designed to confuse parsers are caught at the
    text level.
    """
    for pattern, message in _INJECTION_PATTERNS:
        if pattern.search(sql):
            return message
    return None


def _find_forbidden_node(
    ast: exp.Expression,
) -> exp.Expression | None:
    """
    Walks the AST looking for any node whose type is in
    _FORBIDDEN_NODE_TYPES.

    Returns the first found node (for error message construction), or
    None if the AST is clean.

    Why check the AST for forbidden nodes when we already checked the
    statement type above? Because a CTE can embed a statement type
    that isn't the outer statement type. For example:

      WITH x AS (DELETE FROM t RETURNING id) SELECT * FROM x

    The outer statement is a Select (passes the type check), but the
    CTE body contains a Delete. This walk catches it.
    """
    for node_type in _FORBIDDEN_NODE_TYPES:
        found = ast.find(node_type)
        if found is not None:
            return found
    return None


def _extract_cte_aliases(ast: exp.Expression) -> set[str]:
    """
    Returns the set of CTE alias names (e.g. "dept_stats" in
    WITH dept_stats AS (...)).

    These appear as exp.Table nodes in the AST (the main query
    references the CTE by name as if it were a table), but they are
    not real database tables. Excluding them prevents false-positive
    "hallucinated table" rejections on every CTE query.
    """
    return {cte.alias for cte in ast.find_all(exp.CTE) if cte.alias}


def _build_alias_map(ast: exp.Expression) -> dict[str, str]:
    """
    Builds a mapping from every table alias to its real table name,
    plus each real table name to itself (for unaliased references).

    Example: "FROM employees AS e" → {"e": "employees", "employees": "employees"}

    Required for column validation: column references through aliases
    (e.name) have col.table == "e", not "employees". Without this map,
    every aliased column reference would fail schema validation as an
    unknown table.
    """
    alias_map: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        real_name = table.name
        if not real_name:
            continue
        alias_map[real_name] = real_name
        if table.alias:
            alias_map[table.alias] = real_name
    return alias_map


def _extract_real_tables(
    ast: exp.Expression,
    cte_aliases: set[str],
) -> set[str]:
    """
    Returns the set of real (non-CTE-alias) table names referenced in
    the AST.

    Filters out:
      - CTE alias names (they look like tables but aren't)
      - Empty names (defensive; shouldn't occur from valid SQL)
    """
    return {
        table.name
        for table in ast.find_all(exp.Table)
        if table.name and table.name not in cte_aliases
    }


def _extract_column_refs(
    ast: exp.Expression,
    alias_map: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Returns a list of (resolved_table_name, column_name) tuples for
    every column reference in the AST.

    Alias resolution: col.table may be an alias ("e") rather than the
    real table name ("employees"). alias_map is used to resolve it.

    Unqualified references (col.table is empty — no table prefix on
    the column) are recorded as ("", col_name). These are NOT
    validated against the schema because the correct table is
    ambiguous without query planning — they fall through to the
    database engine itself, which will raise an error if the column
    truly doesn't exist in any in-scope table. Only qualified
    references are validated here.

    Special cases excluded from the result:
      - The wildcard (*): not a column name, never validated.
      - Numeric literals appearing as column references (from
        ORDER BY 1 style clauses): filtered by digit check.
    """
    refs: list[tuple[str, str]] = []
    for col in ast.find_all(exp.Column):
        col_name = col.name
        if not col_name or col_name == "*" or col_name.isdigit():
            continue
        raw_table = col.table or ""
        resolved_table = alias_map.get(raw_table, raw_table)
        refs.append((resolved_table, col_name))
    return refs


def _format_column_refs(refs: list[tuple[str, str]]) -> list[str]:
    """
    Converts (table, column) tuples to "table.column" strings for
    ValidationResult.detected_columns. Unqualified references become
    ".column" (the empty table prefix is preserved as a signal that
    these weren't resolvable at validation time).
    """
    seen: set[str] = set()
    result: list[str] = []
    for table, col in refs:
        key = f"{table}.{col}" if table else f".{col}"
        if key not in seen:
            seen.add(key)
            result.append(key)
    return sorted(result)


def _check_access_policy(
    real_tables: set[str],
    rules: SecurityRules,
) -> str | None:
    """
    Enforces the schema_access policy from security_rules.yaml:

    allow_all mode: all tables are accessible; only denied_tables
        and denied_columns apply (column denial is enforced at the
        query_executor level since column-level denial requires
        knowing which columns appear in SELECT *, which requires
        execution context not available here).

    explicit_allow_list mode: only tables in allowed_tables may be
        queried; anything else is rejected.

    deny_list mode: tables in denied_tables are rejected; all others
        are allowed.

    Returns an error message string if access is denied, else None.
    """
    mode = rules.schema_access.mode

    if mode == "explicit_allow_list" and rules.schema_access.allowed_tables:
        disallowed = real_tables - rules.schema_access.allowed_tables
        if disallowed:
            return (
                f"Access to table(s) {sorted(disallowed)} is not permitted. "
                f"Only explicitly allowed tables may be queried."
            )

    elif mode == "deny_list" and rules.schema_access.denied_tables:
        blocked = real_tables & rules.schema_access.denied_tables
        if blocked:
            return (
                f"Access to table(s) {sorted(blocked)} is not permitted."
            )

    return None


def _check_tables_in_schema(
    real_tables: set[str],
    schema: DatabaseSchema,
) -> str | None:
    """
    Verifies that every real table referenced in the SQL exists in
    the discovered schema.

    Comparison is case-insensitive: the LLM may capitalise table names
    differently from how SQLite stores them ("Employees" vs "employees").
    The error message names the specific hallucinated table and lists
    the available tables, giving both the LLM (retry prompt) and the
    developer (logs) enough information to understand the failure.

    Returns an error message string if any table is not in the schema,
    else None.
    """
    schema_table_names_lower = {t.lower(): t for t in schema.table_names}

    for table in real_tables:
        if table.lower() not in schema_table_names_lower:
            available = ", ".join(sorted(schema.table_names))
            return (
                f"Table '{table}' does not exist in the database. "
                f"Available tables: {available}."
            )

    return None


def _check_columns_in_schema(
    column_refs: list[tuple[str, str]],
    schema: DatabaseSchema,
) -> str | None:
    """
    Verifies that every qualified column reference (table.column) in
    the SQL exists in the discovered schema for that specific table.

    Unqualified references (empty table prefix — see
    _extract_column_refs docstring) are skipped here; they can't be
    validated without query-execution context.

    Comparison is case-insensitive for both table and column names.

    Returns an error message string on the first bad reference found,
    else None.
    """
    for table_name, col_name in column_refs:
        if not table_name:
            continue  # unqualified — skip

        table_info = next(
            (t for t in schema.tables if t.name.lower() == table_name.lower()),
            None,
        )
        if table_info is None:
            # Table not in schema — caught by _check_tables_in_schema already
            # if it's a real table reference. Skip to avoid a redundant error.
            continue

        col_names_lower = {c.name.lower() for c in table_info.columns}
        if col_name.lower() not in col_names_lower:
            available = ", ".join(sorted(c.name for c in table_info.columns))
            return (
                f"Column '{col_name}' does not exist in table '{table_info.name}'. "
                f"Available columns: {available}."
            )

    return None


def _enforce_limit(
    ast: exp.Expression,
    max_rows: int,
    dialect: str,
) -> tuple[exp.Expression, bool]:
    """
    Enforces the row cap on the AST:

    - If no LIMIT is present: injects LIMIT max_rows and returns
      (modified_ast, True).
    - If LIMIT is present and <= max_rows: returns (ast, False)
      unchanged.
    - If LIMIT is present and > max_rows: replaces with LIMIT max_rows
      and returns (modified_ast, True).

    The limit is applied to the outermost SELECT, not to subqueries —
    subquery limits are a query-correctness concern (the LLM should
    write them correctly) rather than a safety concern (the outermost
    result set is what reaches the user).

    For aggregate queries (queries with a GROUP BY clause), a LIMIT
    is not injected if absent — a "SELECT department, AVG(salary)
    GROUP BY department" query returning 5 rows for 5 departments
    should not be capped at 1000 rows as if it were a large table
    scan. The LIMIT injection is skipped for aggregates but an
    existing one is still capped for safety.
    """
    modified = False
    existing_limit = ast.find(exp.Limit)
    has_group_by = ast.find(exp.Group) is not None

    if existing_limit is not None:
        try:
            current_val = int(existing_limit.expression.this)
            if current_val > max_rows:
                ast = ast.limit(max_rows)
                modified = True
        except (AttributeError, ValueError, TypeError):
            # Can't parse the existing limit value — inject a safe cap.
            ast = ast.limit(max_rows)
            modified = True
    elif not has_group_by:
        ast = ast.limit(max_rows)
        modified = True

    return ast, modified


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _reject(
    raw_sql: str,
    error_message: str,
    category: str,
    detected_tables: list[str] | None = None,
    detected_columns: list[str] | None = None,
) -> ValidationResult:
    """Constructs a failed ValidationResult with consistent field values."""
    return ValidationResult(
        is_valid=False,
        original_sql=raw_sql,
        cleaned_sql=None,
        error_message=error_message,
        rejection_category=category,
        detected_tables=detected_tables or [],
        detected_columns=detected_columns or [],
    )
