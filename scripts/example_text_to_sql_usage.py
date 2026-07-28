"""
scripts/example_text_to_sql_usage.py
========================================

Demonstrates the complete Text-to-SQL pipeline:
  DB Connection → Schema Discovery → Schema Context → SQL Generation

Run:
    # With a real Groq API key:
    GROQ_API_KEY=gsk_your_real_key PYTHONPATH=. python3 scripts/example_text_to_sql_usage.py

    # With a fake key (shows pipeline structure; generation will fail at API call):
    GROQ_API_KEY=fake PYTHONPATH=. python3 scripts/example_text_to_sql_usage.py --dry-run

Uses the five example questions from the requirements brief to show
what kinds of SQL the engine is designed to generate.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from core.llm.base_provider import BaseLLMProvider, LLMResponse
from core.nl2sql.sql_generator import SQLGenerationResult, generate_sql
from core.schema.schema_context_builder import build_schema_context
from core.schema.schema_introspector import introspect_schema


def _create_demo_schema():
    """Creates an in-memory database matching the requirements examples."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL,
                manager_id INTEGER,
                hire_date TEXT,
                FOREIGN KEY (manager_id) REFERENCES employees(id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL,
                category TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                product_id INTEGER NOT NULL,
                employee_id INTEGER NOT NULL,
                sale_date TEXT NOT NULL,
                amount REAL NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(id),
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        """))
        conn.commit()
    return engine


# ---------------------------------------------------------------------------
# Pre-scripted "expected" responses — what a well-prompted Groq should
# return for each example question. Used in dry-run mode.
# ---------------------------------------------------------------------------

EXPECTED_RESPONSES = {
    "Show all employees earning more than 80000": {
        "sql": "SELECT *\nFROM employees\nWHERE salary > 80000\nORDER BY salary DESC\nLIMIT 100",
        "reasoning": "Simple filter on the employees table using the salary column with a WHERE clause. ORDER BY salary DESC shows highest earners first.",
        "complexity_tier": "basic",
    },
    "Show the average salary department wise": {
        "sql": "SELECT department,\n       AVG(salary) AS avg_salary\nFROM employees\nGROUP BY department\nORDER BY avg_salary DESC",
        "reasoning": "Aggregation query using AVG with GROUP BY on the department column. No LIMIT needed since this is an aggregate returning one row per department.",
        "complexity_tier": "intermediate",
    },
    "Rank employees according to salary": {
        "sql": (
            "SELECT name,\n"
            "       department,\n"
            "       salary,\n"
            "       RANK() OVER (ORDER BY salary DESC) AS salary_rank\n"
            "FROM employees\n"
            "ORDER BY salary_rank"
        ),
        "reasoning": "Window function RANK() assigns a rank to each employee based on salary descending. RANK() (not DENSE_RANK or ROW_NUMBER) is appropriate because ties should share a rank with a gap after them, which matches the typical meaning of 'ranking'.",
        "complexity_tier": "advanced",
    },
    "Show products that were never sold": {
        "sql": (
            "SELECT p.id,\n"
            "       p.name,\n"
            "       p.price,\n"
            "       p.category\n"
            "FROM products p\n"
            "LEFT JOIN sales s ON p.id = s.product_id\n"
            "WHERE s.id IS NULL\n"
            "ORDER BY p.name"
        ),
        "reasoning": "LEFT JOIN from products to sales, then filter WHERE sales.id IS NULL, which keeps only products with no matching sales row — the classic 'never sold' pattern. A NOT IN subquery would also work but LEFT JOIN + IS NULL is more readable and typically more efficient.",
        "complexity_tier": "intermediate",
    },
    "Show monthly sales for the last year": {
        "sql": (
            "SELECT strftime('%Y-%m', sale_date) AS month,\n"
            "       SUM(amount) AS total_sales,\n"
            "       COUNT(*) AS transaction_count\n"
            "FROM sales\n"
            "WHERE sale_date >= date('now', '-1 year')\n"
            "GROUP BY month\n"
            "ORDER BY month"
        ),
        "reasoning": "SQLite's strftime('%Y-%m', ...) extracts year-month for monthly grouping. date('now', '-1 year') filters to the trailing 12 months. SUM(amount) gives revenue and COUNT(*) gives volume — both useful for a monthly sales view.",
        "complexity_tier": "intermediate",
    },
}


def run_example(
    question: str,
    schema_context,
    dry_run: bool,
) -> None:
    """Runs one question through the full pipeline, printing results."""
    print(f"\n{'='*70}")
    print(f"QUESTION: {question}")
    print('='*70)

    if dry_run:
        expected = EXPECTED_RESPONSES[question]
        fake_payload = json.dumps(expected)

        fake_provider = MagicMock(spec=BaseLLMProvider)
        fake_provider.generate.return_value = LLMResponse(
            content=fake_payload,
            raw_provider_response=None,
            input_tokens=250,
            output_tokens=80,
            latency_ms=55.0,
            model="llama-3.3-70b-versatile (mocked)",
        )

        with patch("core.nl2sql.sql_generator.get_llm_provider") as mock_factory:
            mock_factory.return_value = fake_provider
            result: SQLGenerationResult = generate_sql(question, schema_context)
    else:
        result = generate_sql(question, schema_context)

    if result.clarification_request:
        print(f"⚠ CLARIFICATION NEEDED: {result.clarification_request}")
        return

    print(f"\nGENERATED SQL:\n{result.sql}\n")
    print(f"REASONING:     {result.reasoning}")
    print(f"COMPLEXITY:    {result.complexity_tier}")
    print(f"CONFIDENCE:    {result.confidence:.2f}")
    print(f"ATTEMPTS:      {result.attempt_count}")
    print(f"LATENCY:       {result.latency_ms:.1f}ms")


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("Running in DRY-RUN mode — using pre-scripted responses (no API key required)")
    else:
        print("Running with REAL Groq API — requires valid GROQ_API_KEY")

    engine = _create_demo_schema()
    schema = introspect_schema(engine, include_row_counts=False)

    print("\n=== SCHEMA INJECTED INTO PROMPTS ===")
    from core.schema.schema_introspector import build_schema_summary
    print(build_schema_summary(schema))

    for question in EXPECTED_RESPONSES:
        schema_context = build_schema_context(schema, question)
        run_example(question, schema_context, dry_run=dry_run)

    print("\n\nAll examples completed.")


if __name__ == "__main__":
    main()
