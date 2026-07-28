"""
core/memory/conversation_manager.py
======================================

Canonical storage for conversation turns: question, generated SQL,
result metadata, insight summary, timestamp. Streamlit-free — this
is core/ business logic, not UI state, and is unit-testable without
a Streamlit runtime.

WHY THIS EXISTS SEPARATELY FROM app/state/session_state.py's ChatMessage
--------------------------------------------------------------------------
ChatMessage (session_state.py) is a rendering object: it carries the
full QueryResult (every row, for the data table) and the full
InsightResult, because the UI needs to redraw the complete turn.
ConversationTurn here is deliberately lighter — question, SQL, row
count, and a short insight summary string, not the full result set or
full insight payload. That's not an oversight, it's the point: this
object gets serialized into LLM prompts for follow-up resolution, and
prompt token budget is a real constraint. Feeding 500 rows of a
previous query back into every subsequent prompt would be wasteful
and, past a few turns, would blow the context window. ChatMessage and
ConversationTurn solve different problems and should not be merged
into one object.

WHY "CLEAR CHAT" AND "CLEAR CONTEXT" ARE DISTINCT OPERATIONS
------------------------------------------------------------------
clear_all() empties the turn list entirely — used when the user wants
a full reset (paired with wiping the visible chat transcript too).
clear_context() ALSO empties the turn list in this implementation,
because context here IS the turn list — there is no separate "log"
structure. The distinction that matters is what the CALLER does with
each: app/components/sidebar.py's "Clear Chat" button calls
clear_all() AND wipes the visible ChatMessage transcript. Its
"Clear Context" button calls clear_context() (same underlying effect
on this object) but leaves the ChatMessage transcript on screen
untouched. Two methods exist rather than one shared method so the
call sites read clearly and so a future version where "context" and
"full log" genuinely diverge (e.g. an uncapped audit log alongside a
capped context window) has an obvious extension point without a
signature change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from logging_setup.logger import LogCategory, get_logger

logger = get_logger(__name__)

# Requirement: keep only the last 10 conversation turns.
MAX_TURNS = 10


@dataclass
class ConversationTurn:
    """
    One stored turn: question, generated SQL, result metadata, and a
    short insight summary. Deliberately lightweight — see module
    docstring for why this is not the same object as ChatMessage.
    """

    question: str
    sql: str | None
    row_count: int | None
    execution_time_ms: float | None
    insight_summary: str | None       # InsightResult.summary only, not the full object
    success: bool
    error_message: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationManager:
    """
    Holds up to MAX_TURNS ConversationTurn objects for one session, in
    chronological order (oldest first). Not a singleton — one instance
    per Streamlit session, held in st.session_state by
    app/state/session_state.py, which is the only place that
    constructs or stores this.
    """

    def __init__(self) -> None:
        self._turns: list[ConversationTurn] = []

    def add_turn(self, turn: ConversationTurn) -> None:
        """
        Appends a turn and truncates to the most recent MAX_TURNS.
        Truncation happens here, unconditionally, so no caller can
        forget to cap the list — this is the one place that
        invariant is enforced.
        """
        self._turns.append(turn)
        if len(self._turns) > MAX_TURNS:
            self._turns = self._turns[-MAX_TURNS:]

        logger.debug(
            "conversation_turn_added",
            category=LogCategory.USER_ACTION,
            turn_count=len(self._turns),
            success=turn.success,
        )

    def get_turns(self, limit: int | None = None) -> list[ConversationTurn]:
        """
        Returns stored turns, oldest first. If limit is given, returns
        only the most recent `limit` turns (still oldest-first order)
        — used by context_builder.py to further restrict how many
        turns go into a single prompt, independent of how many are
        retained in memory overall.
        """
        if limit is None:
            return list(self._turns)
        return list(self._turns[-limit:])

    def clear_all(self) -> None:
        """Full reset. Called when the user clears the entire chat —
        both the visible transcript and the AI's memory of it."""
        count = len(self._turns)
        self._turns = []
        logger.info("conversation_cleared_all", category=LogCategory.USER_ACTION, turns_cleared=count)

    def clear_context(self) -> None:
        """
        Empties the turn list used for follow-up resolution, without
        the caller being required to also clear anything visible.
        Functionally identical to clear_all() on this object today —
        see module docstring on why these are still two distinct
        methods rather than one.
        """
        count = len(self._turns)
        self._turns = []
        logger.info("conversation_context_cleared", category=LogCategory.USER_ACTION, turns_cleared=count)

    def __len__(self) -> int:
        return len(self._turns)
