"""
tests/unit/core/test_conversation_memory.py
==============================================

Tests for core/memory/conversation_manager.py and
core/memory/context_builder.py. Both are Streamlit-free, so these
run without any Streamlit runtime or session_state mocking.
"""

from __future__ import annotations


from core.memory.context_builder import build_conversation_context
from core.memory.conversation_manager import (
    MAX_TURNS,
    ConversationManager,
    ConversationTurn,
)


def make_turn(question: str, sql: str | None = "SELECT 1", success: bool = True) -> ConversationTurn:
    return ConversationTurn(
        question=question,
        sql=sql,
        row_count=5,
        execution_time_ms=12.0,
        insight_summary="A short insight." if success else None,
        success=success,
        error_message=None if success else "validation failed",
    )


# ── ConversationManager ─────────────────────────────────────────────────────


class TestConversationManagerBasics:
    def test_starts_empty(self):
        manager = ConversationManager()
        assert len(manager) == 0
        assert manager.get_turns() == []

    def test_add_turn_appends(self):
        manager = ConversationManager()
        manager.add_turn(make_turn("Show sales"))
        assert len(manager) == 1
        assert manager.get_turns()[0].question == "Show sales"

    def test_turns_stored_in_chronological_order(self):
        manager = ConversationManager()
        manager.add_turn(make_turn("first"))
        manager.add_turn(make_turn("second"))
        manager.add_turn(make_turn("third"))
        questions = [t.question for t in manager.get_turns()]
        assert questions == ["first", "second", "third"]

    def test_failed_turns_are_stored_too(self):
        manager = ConversationManager()
        manager.add_turn(make_turn("bad question", sql=None, success=False))
        turn = manager.get_turns()[0]
        assert turn.success is False
        assert turn.sql is None
        assert turn.error_message == "validation failed"


class TestConversationManagerCapping:
    def test_caps_at_max_turns(self):
        manager = ConversationManager()
        for i in range(MAX_TURNS + 5):
            manager.add_turn(make_turn(f"question {i}"))
        assert len(manager) == MAX_TURNS

    def test_caps_keep_most_recent(self):
        """When capped, the OLDEST turns are dropped, not the newest —
        requirement 3 says 'keep only the last 10', meaning the most
        recent ones survive."""
        manager = ConversationManager()
        for i in range(MAX_TURNS + 3):
            manager.add_turn(make_turn(f"question {i}"))
        questions = [t.question for t in manager.get_turns()]
        # First 3 (question 0, 1, 2) should have been dropped.
        assert "question 0" not in questions
        assert "question 1" not in questions
        assert "question 2" not in questions
        assert "question 3" in questions
        assert f"question {MAX_TURNS + 2}" in questions

    def test_get_turns_with_limit(self):
        manager = ConversationManager()
        for i in range(5):
            manager.add_turn(make_turn(f"q{i}"))
        limited = manager.get_turns(limit=2)
        assert len(limited) == 2
        assert [t.question for t in limited] == ["q3", "q4"]


class TestConversationManagerClearing:
    def test_clear_all_empties_turns(self):
        manager = ConversationManager()
        manager.add_turn(make_turn("q1"))
        manager.add_turn(make_turn("q2"))
        manager.clear_all()
        assert len(manager) == 0

    def test_clear_context_empties_turns(self):
        manager = ConversationManager()
        manager.add_turn(make_turn("q1"))
        manager.clear_context()
        assert len(manager) == 0

    def test_clear_on_empty_manager_does_not_raise(self):
        manager = ConversationManager()
        manager.clear_all()  # should not raise on empty state
        manager.clear_context()
        assert len(manager) == 0


# ── context_builder ──────────────────────────────────────────────────────────


class TestBuildConversationContext:
    def test_empty_turns_returns_empty_context(self):
        assert build_conversation_context([]) == []

    def test_single_turn_produces_user_and_assistant_entries(self):
        turns = [make_turn("Show monthly sales", sql="SELECT month, SUM(amount) FROM sales GROUP BY month")]
        context = build_conversation_context(turns)
        assert context == [
            {"role": "user", "content": "Show monthly sales"},
            {"role": "assistant", "content": "SELECT month, SUM(amount) FROM sales GROUP BY month"},
        ]

    def test_turn_with_no_sql_produces_only_user_entry(self):
        """A turn where clarification was requested (no SQL generated)
        should contribute only the user's question — there's no
        meaningful assistant SQL to show."""
        turns = [make_turn("ambiguous question", sql=None)]
        context = build_conversation_context(turns)
        assert context == [{"role": "user", "content": "ambiguous question"}]

    def test_multiple_turns_preserve_order(self):
        turns = [
            make_turn("Show monthly sales", sql="SELECT month, SUM(amount) FROM sales GROUP BY month"),
            make_turn("Only for 2025", sql="SELECT month, SUM(amount) FROM sales WHERE year=2025 GROUP BY month"),
        ]
        context = build_conversation_context(turns)
        assert len(context) == 4
        assert context[0]["content"] == "Show monthly sales"
        assert context[2]["content"] == "Only for 2025"

    def test_respects_max_turns_limit(self):
        turns = [make_turn(f"q{i}") for i in range(10)]
        context = build_conversation_context(turns, max_turns=3)
        # 3 turns * 2 entries each (user + assistant, since sql defaults non-None)
        assert len(context) == 6
        # Should be the LAST 3 turns, not the first 3.
        user_questions = [e["content"] for e in context if e["role"] == "user"]
        assert user_questions == ["q7", "q8", "q9"]

    def test_max_turns_zero_returns_all(self):
        """max_turns=0 is falsy in Python — the implementation treats
        it as 'no limit' via `if max_turns else turns`. Documenting
        this explicit behavior with a test rather than leaving it as
        an implicit edge case."""
        turns = [make_turn(f"q{i}") for i in range(3)]
        context = build_conversation_context(turns, max_turns=0)
        user_questions = [e["content"] for e in context if e["role"] == "user"]
        assert user_questions == ["q0", "q1", "q2"]

    def test_failed_turn_still_included_as_user_context(self):
        """A prior turn that failed validation is still relevant
        context ('don't repeat that mistake') — it must not be
        silently dropped from the context."""
        turns = [make_turn("bad question", sql=None, success=False)]
        context = build_conversation_context(turns)
        assert context == [{"role": "user", "content": "bad question"}]


class TestFollowUpScenario:
    """
    End-to-end style test matching the exact example from the task:
    'Show monthly sales.' followed by 'Only for 2025.' — verifies the
    context a real follow-up question would receive.
    """

    def test_two_turn_followup_context_shape(self):
        manager = ConversationManager()
        manager.add_turn(ConversationTurn(
            question="Show monthly sales.",
            sql="SELECT strftime('%Y-%m', sale_date) AS month, SUM(amount) FROM sales GROUP BY month",
            row_count=12,
            execution_time_ms=45.0,
            insight_summary="Sales peaked in December.",
            success=True,
        ))

        context = build_conversation_context(manager.get_turns())

        # The next question "Only for 2025." would be sent to
        # generate_sql with this context — verify it contains enough
        # for the model to resolve the follow-up (the prior SQL is
        # present, so it can add a WHERE clause to the existing query
        # shape rather than guessing from scratch).
        assert any("Show monthly sales" in e["content"] for e in context)
        assert any("sale_date" in e["content"] for e in context if e["role"] == "assistant")
