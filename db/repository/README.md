# db/repository/

This folder holds the **repository pattern** implementation for the
app's internal storage (db/models.py).

## Why this folder exists separately from db/models.py

`db/models.py` defines *what* the internal data looks like (ORM
models). This folder defines *how* it's accessed — every read/write
against `ConversationSession`, `ConversationTurn`, and `QueryAuditLog`
goes through a repository class here, never through ad hoc
`session.query(...)` calls scattered across `core/` or `app/`.

This matters for two concrete reasons:

1. **Testability.** `core/orchestration/conversation_manager.py` can
   depend on a `ConversationRepository` interface and be tested with
   an in-memory fake, without spinning up a real database.
2. **Single point of change.** If the internal storage migrates from
   SQLite to Postgres, or a caching layer is added, only this folder
   changes — not every module that happens to need conversation
   history.

## Planned files (implementation phase)

- `conversation_repository.py` — CRUD for `ConversationSession` /
  `ConversationTurn`.
- `audit_repository.py` — append-only writes and queries for
  `QueryAuditLog`. Kept separate from `conversation_repository.py`
  deliberately: audit logging should remain functional and decoupled
  even if conversation-history logic changes, since it's a security
  record, not a UX feature.
- `session_factory.py` — provides scoped SQLAlchemy sessions
  (`sessionmaker`) for the internal database engine specifically,
  distinct from `db/connectors/connection_manager.py`, which manages
  engines for the *user's queried* database.
