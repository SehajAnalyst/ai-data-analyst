"""
db/repository/conversation_repository.py
========================================

Repository for ConversationSession and ConversationTurn.

This module is the ONLY place that reads/writes conversation history
from the application's internal database.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from db.models import (
    ConversationSession,
    ConversationTurn,
)


class ConversationRepository:
    """
    CRUD operations for conversations.
    """

    def __init__(self, session: Session):
        self.session = session

    def create_session(
        self,
        title: str,
        database_url: str,
        llm_provider: str,
        db_dialect: str,
    ) -> ConversationSession:
        """
        Create a new conversation session.
        """

        conversation = ConversationSession(
            id=str(uuid.uuid4()),
            title=title,
            database_url=database_url,
            llm_provider=llm_provider,
            db_dialect=db_dialect,
        )

        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)

        return conversation

    def get_session(
        self,
        conversation_id: str,
    ) -> ConversationSession | None:
        """
        Get one conversation by ID.
        """

        return (
            self.session.query(ConversationSession)
            .filter(
                ConversationSession.id == conversation_id
            )
            .first()
        )

    def list_recent_sessions(
        self,
        limit: int = 10,
    ) -> list[ConversationSession]:
        """
        Return the most recently updated conversations.
        """

        return (
            self.session.query(ConversationSession)
            .order_by(
                ConversationSession.updated_at.desc()
            )
            .limit(limit)
            .all()
        )

    def add_turn(
        self,
        conversation_id: str,
        user_question: str,
        assistant_response: str,
        generated_sql: str | None = None,
        validation_result: str | None = None,
        execution_status: str | None = None,
        row_count: int | None = None,
        insight_text: str | None = None,
        insight_data: dict | None = None,
        chart_metadata: dict | None = None,
    ) -> ConversationTurn:
        """
        Save one conversation turn.
        """

        turn = ConversationTurn(
            session_id=conversation_id,
            trace_id=str(uuid.uuid4()),
            user_question=user_question,
            assistant_response=assistant_response,
            generated_sql=generated_sql,
            validation_result=validation_result,
            execution_status=execution_status,
            row_count=row_count,
            insight_text=insight_text,
            insight_data=insight_data,
            chart_metadata=chart_metadata,
        )

        self.session.add(turn)
        self.session.commit()
        self.session.refresh(turn)

        return turn

    def get_turns(
        self,
        conversation_id: str,
    ) -> list[ConversationTurn]:
        """
        Load every turn of one conversation.
        """

        turns = (
            self.session.query(ConversationTurn)
            .filter(
                ConversationTurn.session_id == conversation_id
            )
            .order_by(
                ConversationTurn.created_at.asc()
            )
            .all()
        )

        
        return turns

    def rename_session(
        self,
        conversation_id: str,
        title: str,
    ) -> None:
        """
        Rename a conversation.
        """

        conversation = self.get_session(conversation_id)

        if conversation:
            conversation.title = title
            self.session.commit()

    def delete_session(
        self,
        conversation_id: str,
    ) -> None:
        """
        Delete a conversation and all its turns.
        """

        conversation = self.get_session(conversation_id)

        if conversation:
            self.session.delete(conversation)
            self.session.commit()