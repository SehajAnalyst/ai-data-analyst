"""
core/memory/context_builder.py
=================================

Converts stored ConversationTurn objects into the exact
list[{"role": ..., "content": ...}] shape that
core/nl2sql/sql_generator.generate_sql() already accepts via its
conversation_history parameter.

WHY THIS DOESN'T TOUCH sql_generator.py
------------------------------------------
sql_generator.py is a completed, tested module with its own private
_format_conversation_history() that renders a list[dict] into prompt
text. That split (context_builder decides WHICH turns and WHAT
fields go in; sql_generator's private formatter decides HOW to render
them into prompt text) is legitimate layering, not duplication — this
module owns turn SELECTION, sql_generator owns prompt RENDERING.
Changing sql_generator's input shape would mean re-touching and
re-testing a module that already works; this module conforms to the
existing contract instead.

WHY A SEPARATE MODULE RATHER THAN INLINE IN app/main.py
-----------------------------------------------------------
Before this module existed, app/main.py built this list by hand,
inline, in the Streamlit page — business logic (deciding what context
the LLM sees) sitting in the UI layer, untested, and duplicating
ConversationManager's own knowledge of turn ordering. Moving it here
makes it unit-testable and keeps app/main.py a thin caller.
"""

from __future__ import annotations

from core.memory.conversation_manager import ConversationTurn
from logging_setup.logger import get_logger

logger = get_logger(__name__)

# How many recent turns to include in the LLM prompt. Deliberately
# smaller than MAX_TURNS (10) stored in ConversationManager — token
# budget for follow-up context should stay tight even if more history
# is retained for the sidebar/audit view. 6 turns covers the realistic
# depth of a follow-up chain ("show X" -> "only 2025" -> "sort desc")
# without bloating every subsequent prompt with distant history.
DEFAULT_MAX_CONTEXT_TURNS = 6


def build_conversation_context(
    turns: list[ConversationTurn],
    max_turns: int = DEFAULT_MAX_CONTEXT_TURNS,
) -> list[dict]:
    """
    Converts stored turns into the list[dict] shape
    sql_generator.generate_sql() expects.

    Args:
        turns: full stored turn list from
            ConversationManager.get_turns(), oldest first.
        max_turns: caps how many of the most recent turns are
            included. Failed turns (success=False) are still included
            — a prior turn that failed validation is still relevant
            context ("don't repeat that mistake"), and dropping it
            silently would hide useful signal from the model.

    Returns:
        list of {"role": "user"|"assistant", "content": str} dicts,
        oldest first, ready to pass directly as
        generate_sql(conversation_history=...).

        For each turn: one "user" entry (the question) and, only if
        SQL was actually generated, one "assistant" entry containing
        that SQL. Turns with no SQL (e.g. a clarification was
        requested instead) contribute only the user entry — there is
        nothing meaningful to show as the assistant's turn.
    """
    recent = turns[-max_turns:] if max_turns else turns
    context: list[dict] = []

    for turn in recent:
        context.append({"role": "user", "content": turn.question})
        if turn.sql:
            context.append({"role": "assistant", "content": turn.sql})

    logger.debug(
        "conversation_context_built",
        turns_considered=len(recent),
        context_entries=len(context),
    )

    return context
