# core/nl2sql/few_shot_examples/

Curated (question, schema, SQL) examples used in SQL generation
prompts, tiered by complexity. Stored as data files (JSON/YAML), not
Python, for the same reason prompt_templates/ uses .txt files — these
are content to be iterated on, not logic.

## Planned structure (populated during implementation)

- `basic_examples.json` — SELECT/WHERE/ORDER BY/LIMIT, simple filters
- `intermediate_examples.json` — aggregations, GROUP BY/HAVING, JOINs,
  CASE, date functions
- `advanced_examples.json` — CTEs, window functions, correlated
  subqueries, self-joins, UNION, pivoting

## Why tiered rather than one flat example pool

Per the architecture doc (section 8): including advanced examples for
a simple "how many customers do we have" question wastes tokens and
can bias the model toward overcomplicating simple queries. The
question's apparent complexity (assessed lightly in sql_generator.py)
determines which tier(s) of examples get pulled into the prompt.

Each example should include the schema it was generated against, so
examples remain meaningful out of context and so prompt assembly can
optionally pick examples whose schema shape resembles the current
question's relevant tables.
