"""
db/connectors/postgres_connector.py
======================================

PostgreSQL implementation of BaseDBConnector. Planned for Phase 2
(see roadmap) — stubbed now so the connector interface is proven out
across more than one dialect from the start, even though SQLite is the
only one wired up in V1.

NOTE ON read_only: for Postgres, `read_only=True` must map to
connecting as a dedicated read-only database role (e.g. one granted
only SELECT on relevant schemas), not just an app-level flag. The
connection string/credentials used here should point at that
restricted role specifically — this connector should not assume the
app-level validator is sufficient on its own (see
config/security_rules.yaml header comment on defense-in-depth).
"""

from sqlalchemy.engine import Engine

from db.connectors.base_connector import BaseDBConnector


class PostgresConnector(BaseDBConnector):
    @property
    def dialect_name(self) -> str:
        return "postgresql"

    # create_engine() / test_connection() implementation deferred to Phase 2.
