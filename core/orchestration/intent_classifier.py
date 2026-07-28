"""
core/orchestration/intent_classifier.py
==========================================

Classifies each incoming user message before any SQL generation
happens, so the orchestrator knows which pipeline to route to.

WHY THIS RUNS BEFORE SQL GENERATION, AS ITS OWN STEP
-----------------------------------------------------------
Not every message is a new data question. Per the architecture doc's
workflow (step 1), a message could be:
  - a brand-new question requiring fresh schema context
  - a follow-up modifying the previous query ("now break it down by
    month") — needs prior SQL/context, not a fresh schema lookup
  - an ML request ("forecast next quarter's revenue") — routes to
    ml_plugins/, not core/nl2sql/ at all
  - out of scope ("write me a poem") — should be declined cleanly,
    not forced through SQL generation where the LLM might produce
    nonsense SQL just to comply

Without this classification step up front, every message gets the
same expensive treatment (full schema context assembly + SQL
generation) regardless of whether that's appropriate, and out-of-
scope or ambiguous messages produce confusing or wasteful SQL-shaped
output instead of a clean response.

DESIGN NOTE: this can be a cheap/fast LLM call (a good candidate for
the Groq provider specifically, per the note in
core/llm/providers/groq_provider.py, regardless of which provider the
user has selected as their default for SQL generation) since it's a
simple classification task, not complex reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    NEW_QUERY = "new_query"
    FOLLOW_UP = "follow_up"
    ML_REQUEST = "ml_request"          # routes to ml_plugins/
    CLARIFICATION_NEEDED = "clarification_needed"  # ambiguous, ask the user
    OUT_OF_SCOPE = "out_of_scope"      # politely decline


@dataclass
class IntentClassification:
    intent: IntentType
    confidence: float
    # Populated only when intent == ML_REQUEST, used to route to the
    # correct ml_plugins/ plugin (see ml_plugins/plugin_registry.py).
    suggested_ml_capability: str | None = None


def classify_intent(
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> IntentClassification:
    """
    Classifies the user's message into one of IntentType.

    Args:
        user_message: current message text.
        conversation_history: recent turns — necessary because
            follow-up detection inherently depends on what was asked
            before ("now by region" is only classifiable as
            FOLLOW_UP in light of a prior NEW_QUERY turn).

    Returns:
        IntentClassification, used by
        core/orchestration/conversation_manager.py to route to
        core/nl2sql/, ml_plugins/, or a clarification/decline response.

    Raises:
        IntentClassificationError: if classification confidence is
            too low to act on confidently — caller should treat this
            similarly to CLARIFICATION_NEEDED.

    Implementation deferred to implementation phase.
    """
    raise NotImplementedError("Intent classification pending implementation phase.")
