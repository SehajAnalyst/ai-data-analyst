"""
db/connectors/mysql_connector.py
===================================

MySQL implementation of BaseDBConnector. Planned for Phase 2 — see
postgres_connector.py for the same read_only design note, which
applies equally here (read-only MySQL user/role, not just app-level
filtering).
"""

from sqlalchemy.engine import Engine

from db.connectors.base_connector import BaseDBConnector


class MySQLConnector(BaseDBConnector):
    @property
    def dialect_name(self) -> str:
        return "mysql"

    # create_engine() / test_connection() implementation deferred to Phase 2.
