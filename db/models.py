"""
db/models.py
=============

SQLAlchemy ORM models for the application's OWN internal storage —
conversation history, query logs, saved settings.

CRITICAL DISTINCTION FROM THE REST OF db/
----------------------------------------------
This is the ONE place in the project where SQLAlchemy's declarative
ORM is used with known, fixed models. Everywhere else (the user's
connected database being queried), we use SQLAlchemy Core +
reflection, because that schema is arbitrary and not known in advance.

Conflating these two would be a mistake: the user's database is
untrusted, dynamic, and read-only from our perspective. The app's
internal database is ours, has a fixed schema we control via
migrations, and is read-write. Mixing them under one model risks,
at best, confusing code, and at worst, accidentally pointing
write operations at a customer's production database.

This module defines models only; it has no engine/session logic
(that lives in db/repository/, which is the only code permitted to
import these models and issue queries against them — see
repository/README note below).
"""

from __future__ import annotations

import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all internal app models."""


class ConversationSession(Base):
    """One persistent conversation."""

    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    title: Mapped[str] = mapped_column(String(200))

    database_url: Mapped[str] = mapped_column(Text)

    llm_provider: Mapped[str] = mapped_column(String(50))

    db_dialect: Mapped[str] = mapped_column(String(50))

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    turns: Mapped[list["ConversationTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )

class ConversationTurn(Base):
    """
    One question/answer turn within a session. This is the persistence
    backing for conversation memory if/when it needs to survive beyond
    a single Streamlit session (st.session_state alone is in-memory
    only and does not persist across restarts — see architecture doc
    section 6 for that caveat).
    """

    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversation_sessions.id"))
    trace_id: Mapped[str] = mapped_column(String(36))  # correlates with logging_setup trace_id
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    user_question: Mapped[str] = mapped_column(Text)

    assistant_response: Mapped[str] = mapped_column(Text)

    generated_sql: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    validation_result: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    execution_status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    insight_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    chart_metadata: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    session: Mapped["ConversationSession"] = relationship(back_populates="turns")


class QueryAuditLog(Base):
    """
    Append-only audit record of every SQL execution attempt, including
    REJECTED ones. This is distinct from ConversationTurn (which is
    about UX/conversation continuity) — this table exists specifically
    for security auditing: "show me every query that was rejected by
    the validator in the last 30 days" should be answerable from this
    table alone, without touching conversation data.
    """

    __tablename__ = "query_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    sql_text: Mapped[str] = mapped_column(Text)
    validation_result: Mapped[str] = mapped_column(String(20))
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    executed: Mapped[bool] = mapped_column(default=False)
