"""
tests/sql_validator_test_suite/test_sql_validator.py
=======================================================

Adversarial test suite for core/nl2sql/sql_validator.py.

DESIGN: ADVERSARIAL, NOT JUST HAPPY-PATH
------------------------------------------
This test file exists specifically in tests/sql_validator_test_suite/
(not tests/unit/) because it serves a different purpose: it is an
adversarial corpus, not a unit test suite. Every test here represents
either a known attack pattern, a discovered edge case, or a class of
input that an AI model might plausibly generate incorrectly.

Adding new tests here when you discover a new attack vector or a
validator bug is the expected workflow — this file should grow over
time, not stay fixed. Each test should document WHY the case is
interesting, not just assert a result.

USES A REAL SCHEMA AND REAL RULES
-----------------------------------
Unlike tests/unit/, these tests use real SecurityRules (loaded from
config/security_rules.yaml) and a real introspected DatabaseSchema
(from an in-memory SQLite database). The validator's entire value is
in how it behaves against real inputs; mocking the schema or rules
would test that the validator calls its dependencies correctly, not
that it actually catches real attacks against a real schema.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from core.nl2sql.security_rules_loader import SecurityRules, SchemaAccessConfig
from core.nl2sql.sql_validator import validate_sql
from core.schema.schema_introspector import introspect_schema


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def schema():
    """
    Real in-memory schema with employees, departments, and a sales
    table (to enable JOIN and FK tests). Module-scoped: building the
    schema once per test module is sufficient since the schema is not
    mutated by any test.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE employees ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  department TEXT NOT NULL,"
            "  salary REAL NOT NULL,"
            "  manager_id INTEGER"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE departments ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  budget REAL"
            ")"
        ))
        conn.execute(text(
            "CREATE TABLE sales ("
            "  id INTEGER PRIMARY KEY,"
            "  employee_id INTEGER NOT NULL,"
            "  amount REAL NOT NULL,"
            "  sale_date TEXT NOT NULL"
            ")"
        ))
        conn.commit()
    return introspect_schema(engine, include_row_counts=False)


@pytest.fixture(scope="module")
def rules():
    from core.nl2sql.security_rules_loader import load_security_rules
    return load_security_rules()


# ─── Valid SQL: all of these MUST pass ───────────────────────────────────────


class TestValidSQL:
    def test_basic_select_all(self, schema, rules):
        r = validate_sql("SELECT * FROM employees", schema, rules=rules)
        assert r.is_valid

    def test_basic_select_with_where(self, schema, rules):
        r = validate_sql(
            "SELECT name, salary FROM employees WHERE salary > 80000", schema, rules=rules
        )
        assert r.is_valid
        assert "employees" in r.detected_tables

    def test_limit_injected_when_absent(self, schema, rules):
        r = validate_sql("SELECT * FROM employees", schema, rules=rules)
        assert r.is_valid
        assert "LIMIT" in r.cleaned_sql.upper()

    def test_limit_preserved_when_present_and_within_cap(self, schema, rules):
        r = validate_sql("SELECT * FROM employees LIMIT 10", schema, rules=rules)
        assert r.is_valid
        assert "LIMIT 10" in r.cleaned_sql

    def test_oversized_limit_capped_to_max(self, schema, rules):
        r = validate_sql("SELECT * FROM employees LIMIT 9999999", schema, rules=rules)
        assert r.is_valid
        assert "LIMIT 1000" in r.cleaned_sql
        assert "9999999" not in r.cleaned_sql

    def test_aggregate_no_limit_injected(self, schema, rules):
        """GROUP BY queries return one row per group — injecting LIMIT
        1000 on a 5-department query is technically safe but misleading.
        The validator skips LIMIT injection for aggregate queries."""
        r = validate_sql(
            "SELECT department, AVG(salary) FROM employees GROUP BY department",
            schema, rules=rules,
        )
        assert r.is_valid
        assert "LIMIT" not in r.cleaned_sql.upper()

    def test_inner_join(self, schema, rules):
        r = validate_sql(
            "SELECT e.name, d.budget FROM employees e "
            "INNER JOIN departments d ON e.department = d.name",
            schema, rules=rules,
        )
        assert r.is_valid
        assert set(r.detected_tables) == {"employees", "departments"}

    def test_left_join(self, schema, rules):
        r = validate_sql(
            "SELECT e.name, s.amount FROM employees e "
            "LEFT JOIN sales s ON e.id = s.employee_id",
            schema, rules=rules,
        )
        assert r.is_valid

    def test_cte_does_not_false_positive_on_alias(self, schema, rules):
        """'dept_stats' is a CTE alias, not a real table. The validator
        must NOT flag it as a hallucinated table."""
        r = validate_sql(
            "WITH dept_stats AS ("
            "  SELECT department, AVG(salary) AS avg FROM employees GROUP BY department"
            ") SELECT * FROM dept_stats WHERE avg > 50000",
            schema, rules=rules,
        )
        assert r.is_valid
        # The CTE alias should not appear in detected_tables
        assert "dept_stats" not in r.detected_tables
        assert "employees" in r.detected_tables

    def test_window_function_with_table_alias(self, schema, rules):
        """Column references through aliases (e.salary) must resolve
        correctly — the validator must not reject 'salary' as not found
        in table 'e' (which isn't in the schema)."""
        r = validate_sql(
            "SELECT e.name, RANK() OVER (ORDER BY e.salary DESC) AS rnk "
            "FROM employees AS e",
            schema, rules=rules,
        )
        assert r.is_valid

    def test_subquery(self, schema, rules):
        r = validate_sql(
            "SELECT name, salary FROM employees "
            "WHERE salary > (SELECT AVG(salary) FROM employees)",
            schema, rules=rules,
        )
        assert r.is_valid

    def test_union_all(self, schema, rules):
        r = validate_sql(
            "SELECT id, name FROM employees UNION ALL SELECT id, name FROM departments",
            schema, rules=rules,
        )
        assert r.is_valid

    def test_self_join(self, schema, rules):
        r = validate_sql(
            "SELECT e.name, m.name AS manager "
            "FROM employees e LEFT JOIN employees m ON e.manager_id = m.id",
            schema, rules=rules,
        )
        assert r.is_valid
        # employees appears once in detected_tables even in a self-join
        assert "employees" in r.detected_tables

    def test_case_expression(self, schema, rules):
        r = validate_sql(
            "SELECT name, CASE WHEN salary > 100000 THEN 'high' "
            "WHEN salary > 50000 THEN 'mid' ELSE 'low' END AS tier "
            "FROM employees",
            schema, rules=rules,
        )
        assert r.is_valid

    def test_or_and_not_where(self, schema, rules):
        r = validate_sql(
            "SELECT * FROM employees WHERE NOT (department = 'HR' OR salary < 30000)",
            schema, rules=rules,
        )
        assert r.is_valid

    def test_original_sql_preserved_unchanged(self, schema, rules):
        sql = "SELECT name FROM employees WHERE salary > 50000"
        r = validate_sql(sql, schema, rules=rules)
        assert r.original_sql == sql

    def test_cleaned_sql_differs_when_limit_injected(self, schema, rules):
        sql = "SELECT name FROM employees"
        r = validate_sql(sql, schema, rules=rules)
        assert r.original_sql == sql
        assert r.cleaned_sql != sql
        assert "LIMIT" in r.cleaned_sql.upper()


# ─── Dangerous statements: all MUST be rejected ──────────────────────────────


class TestDangerousStatements:
    @pytest.mark.parametrize("sql,label", [
        ("DROP TABLE employees", "DROP"),
        ("DELETE FROM employees WHERE id = 1", "DELETE"),
        ("INSERT INTO employees VALUES (1,'x','y',1,NULL)", "INSERT"),
        ("UPDATE employees SET salary = 0", "UPDATE"),
        ("ALTER TABLE employees ADD COLUMN foo TEXT", "ALTER"),
        ("TRUNCATE TABLE employees", "TRUNCATE"),
        ("CREATE TABLE foo (id INTEGER)", "CREATE"),
        ("GRANT SELECT ON employees TO user1", "GRANT"),
        ("REVOKE SELECT ON employees FROM user1", "REVOKE"),
    ])
    def test_rejects_dangerous_statement(self, sql, label, schema, rules):
        r = validate_sql(sql, schema, rules=rules)
        assert not r.is_valid, f"{label} should have been rejected"
        assert r.rejection_category == "safety_violation"

    def test_cte_wrapping_delete_is_caught(self, schema, rules):
        """Wrapping DELETE in a CTE should not bypass the validator.
        The outer statement is a Select but the CTE body contains a
        Delete AST node — the forbidden-node walk catches it."""
        r = validate_sql(
            "WITH del AS (DELETE FROM employees RETURNING id) SELECT * FROM del",
            schema, rules=rules,
        )
        assert not r.is_valid
        assert r.rejection_category == "safety_violation"


# ─── Comment injection: all MUST be rejected ─────────────────────────────────


class TestCommentInjection:
    def test_line_comment_rejected(self, schema, rules):
        r = validate_sql(
            "SELECT * FROM employees -- DROP TABLE employees", schema, rules=rules
        )
        assert not r.is_valid
        assert r.rejection_category == "safety_violation"

    def test_block_comment_rejected(self, schema, rules):
        r = validate_sql(
            "SELECT * FROM employees /* this is evil */ WHERE 1=1", schema, rules=rules
        )
        assert not r.is_valid
        assert r.rejection_category == "safety_violation"

    def test_comment_before_dangerous_keyword_rejected(self, schema, rules):
        r = validate_sql(
            "SELECT 1 /*\nDROP TABLE employees\n*/", schema, rules=rules
        )
        assert not r.is_valid


# ─── Injection patterns ───────────────────────────────────────────────────────


class TestInjectionPatterns:
    def test_sqlite_master_rejected(self, schema, rules):
        r = validate_sql("SELECT * FROM sqlite_master", schema, rules=rules)
        assert not r.is_valid

    def test_sqlite_schema_rejected(self, schema, rules):
        r = validate_sql("SELECT * FROM sqlite_schema", schema, rules=rules)
        assert not r.is_valid

    def test_information_schema_rejected(self, schema, rules):
        r = validate_sql("SELECT * FROM information_schema.tables", schema, rules=rules)
        assert not r.is_valid
        assert r.rejection_category == "injection_pattern"

    def test_xp_proc_rejected(self, schema, rules):
        r = validate_sql(
            "SELECT * FROM employees WHERE name = xp_cmdshell('dir')",
            schema, rules=rules,
        )
        assert not r.is_valid
        assert r.rejection_category == "injection_pattern"

    def test_multiple_statements_rejected(self, schema, rules):
        r = validate_sql(
            "SELECT * FROM employees; DROP TABLE employees", schema, rules=rules
        )
        assert not r.is_valid
        assert r.rejection_category == "safety_violation"
        assert "2" in r.error_message  # error message mentions the statement count

    def test_empty_sql_rejected(self, schema, rules):
        r = validate_sql("", schema, rules=rules)
        assert not r.is_valid
        assert r.rejection_category == "syntax_error"

    def test_whitespace_only_rejected(self, schema, rules):
        r = validate_sql("   \n\t  ", schema, rules=rules)
        assert not r.is_valid


# ─── Hallucination detection ─────────────────────────────────────────────────


class TestHallucinationDetection:
    def test_hallucinated_table_caught(self, schema, rules):
        r = validate_sql("SELECT * FROM payroll WHERE amount > 1000", schema, rules=rules)
        assert not r.is_valid
        assert r.rejection_category == "schema_mismatch"
        assert "payroll" in r.error_message
        # Error message must list valid tables so LLM can self-correct
        assert "employees" in r.error_message

    def test_hallucinated_column_caught(self, schema, rules):
        r = validate_sql(
            "SELECT employees.reveune FROM employees", schema, rules=rules
        )
        assert not r.is_valid
        assert r.rejection_category == "schema_mismatch"
        assert "reveune" in r.error_message
        # Error message must list valid columns
        assert "salary" in r.error_message

    def test_real_table_real_column_passes(self, schema, rules):
        r = validate_sql("SELECT employees.salary FROM employees", schema, rules=rules)
        assert r.is_valid

    def test_case_insensitive_table_match(self, schema, rules):
        """The LLM may capitalise table names differently; the validator
        must not reject valid tables just because of case differences."""
        r = validate_sql("SELECT * FROM EMPLOYEES LIMIT 10", schema, rules=rules)
        assert r.is_valid

    def test_case_insensitive_column_match(self, schema, rules):
        r = validate_sql("SELECT EMPLOYEES.SALARY FROM EMPLOYEES LIMIT 10", schema, rules=rules)
        assert r.is_valid

    def test_unqualified_columns_not_validated(self, schema, rules):
        """Unqualified column references (no table prefix) are skipped
        by the validator — they can't be resolved without execution
        context. This is intentional, not a gap."""
        r = validate_sql(
            "SELECT completely_made_up_column FROM employees", schema, rules=rules
        )
        # Unqualified column → no schema check → passes validation
        assert r.is_valid


# ─── Output shape ────────────────────────────────────────────────────────────


class TestOutputShape:
    def test_detected_tables_populated_on_pass(self, schema, rules):
        r = validate_sql(
            "SELECT e.name FROM employees e JOIN departments d ON e.department = d.name",
            schema, rules=rules,
        )
        assert r.is_valid
        assert set(r.detected_tables) == {"employees", "departments"}

    def test_detected_tables_populated_on_fail(self, schema, rules):
        """Even rejected queries should have detected_tables populated
        where possible — useful for logging what the LLM attempted."""
        r = validate_sql(
            "SELECT * FROM payroll JOIN employees ON payroll.id = employees.id",
            schema, rules=rules,
        )
        assert not r.is_valid
        # employees was detected (it exists); payroll caused the failure
        assert "employees" in r.detected_tables or "payroll" in r.detected_tables

    def test_detected_columns_populated(self, schema, rules):
        r = validate_sql(
            "SELECT employees.name, employees.salary FROM employees",
            schema, rules=rules,
        )
        assert r.is_valid
        assert "employees.name" in r.detected_columns
        assert "employees.salary" in r.detected_columns

    def test_cleaned_sql_is_none_on_failure(self, schema, rules):
        r = validate_sql("DROP TABLE employees", schema, rules=rules)
        assert not r.is_valid
        assert r.cleaned_sql is None

    def test_error_message_is_none_on_success(self, schema, rules):
        r = validate_sql("SELECT * FROM employees LIMIT 1", schema, rules=rules)
        assert r.is_valid
        assert r.error_message is None

    def test_rejection_category_is_none_on_success(self, schema, rules):
        r = validate_sql("SELECT name FROM employees LIMIT 1", schema, rules=rules)
        assert r.is_valid
        assert r.rejection_category is None


# ─── Access policy ────────────────────────────────────────────────────────────


class TestAccessPolicy:
    def test_explicit_allow_list_blocks_unlisted_table(self, schema):
        restricted_rules = SecurityRules(
            allowed_statement_types=frozenset(["SELECT"]),
            forbidden_keywords=frozenset(["DROP", "DELETE"]),
            reject_sql_comments=True,
            reject_multiple_statements=True,
            max_rows_returned=1000,
            schema_access=SchemaAccessConfig(
                mode="explicit_allow_list",
                allowed_tables=frozenset(["employees"]),
                denied_tables=frozenset(),
                denied_columns=frozenset(),
            ),
        )
        r = validate_sql("SELECT * FROM departments", schema, rules=restricted_rules)
        assert not r.is_valid
        assert r.rejection_category == "access_denied"

    def test_explicit_allow_list_permits_listed_table(self, schema):
        restricted_rules = SecurityRules(
            allowed_statement_types=frozenset(["SELECT"]),
            forbidden_keywords=frozenset(["DROP"]),
            reject_sql_comments=True,
            reject_multiple_statements=True,
            max_rows_returned=1000,
            schema_access=SchemaAccessConfig(
                mode="explicit_allow_list",
                allowed_tables=frozenset(["employees"]),
                denied_tables=frozenset(),
                denied_columns=frozenset(),
            ),
        )
        r = validate_sql("SELECT * FROM employees LIMIT 5", schema, rules=restricted_rules)
        assert r.is_valid

    def test_deny_list_blocks_denied_table(self, schema):
        sensitive_rules = SecurityRules(
            allowed_statement_types=frozenset(["SELECT"]),
            forbidden_keywords=frozenset(["DROP"]),
            reject_sql_comments=True,
            reject_multiple_statements=True,
            max_rows_returned=1000,
            schema_access=SchemaAccessConfig(
                mode="deny_list",
                allowed_tables=frozenset(),
                denied_tables=frozenset(["employees"]),
                denied_columns=frozenset(),
            ),
        )
        r = validate_sql("SELECT * FROM employees", schema, rules=sensitive_rules)
        assert not r.is_valid
        assert r.rejection_category == "access_denied"
