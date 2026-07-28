"""
core/orchestration/conversation_manager.py
=============================================

The central orchestrator. This is the ONE module that ties together
intent classification, schema context, SQL generation, validation,
execution, chart selection, and insight generation into a single
handled "turn." Per the architecture doc (section 1), this is what
distinguishes an AI Data Analyst from a bare Text-to-SQL box — it's
also the only module in core/ that's allowed to call across all the
other core/ submodules; everything else interacts through it rather
than reaching directly into siblings (e.g. core/visualization/ should
never call core/llm/ directly — if it needs something from there,
that need flows through this orchestrator).

WHY THIS MODULE EXISTS AS A DISTINCT LAYER ABOVE THE PIPELINE STEPS
-------------------------------------------------------------------------
Without an explicit orchestrator, the natural failure mode is that
app/pages/1_chat.py (the Streamlit page) ends up calling
schema_introspector, then sql_generator, then sql_validator, then
query_executor, then chart_selector, then insight_generator directly,
in sequence, inline in the UI code. That couples business logic to
the UI layer, makes the pipeline untestable without Streamlit, and is
exactly the trap called out in the architecture doc (section 6) about
why core/ must have zero Streamlit dependency. This module is what the
UI calls instead — ONE method, `handle_turn()` — and everything below
it is internal to core/.

STATE MANAGEMENT
------------------
This module is deliberately NOT a stateful singleton holding
conversation history in memory inside the class. Per the architecture
doc's Streamlit-specific constraints (section 6): Streamlit reruns the
whole script on every interaction, so any state held only in a Python
object's `self` attributes would be silently lost on rerun. Instead,
conversation history is passed IN on every call (rehydrated by the
caller from st.session_state or, eventually,
db/repository/conversation_repository.py for persistence beyond a
single session) and the updated history is returned, not mutated
in place. This makes the orchestrator a pure-ish function of
(message, history) -> (response, updated_history), which is also
what makes it straightforward to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnResult:
    """Everything the UI layer needs to render one completed turn."""

    response_text: str            # combined SQL explanation + insight, or a decline/clarification message
    generated_sql: str | None
    sql_explanation: str | None
    chart_selection: object | None    # ChartSelection, see core/visualization/chart_selector.py
    query_result: object | None       # QueryResult, see core/execution/query_executor.py
    insight_text: str | None
    error_message: str | None     # populated, and other fields left None/empty, on failure


def handle_turn(
    user_message: str,
    conversation_history: list[dict],
    session_id: str,
) -> tuple[TurnResult, list[dict]]:
    """
    Handles one full user turn end-to-end: classify intent, route to
    the appropriate pipeline (SQL or ML), execute, and assemble a
    TurnResult ready for display.

    Args:
        user_message: the user's current natural-language input.
        conversation_history: prior turns in this session (read-only
            input; this function does not mutate the list in place —
            see module docstring on state management).
        session_id: correlates logs/audit records for this session
            (see logging_setup.logger.bind_session_context).

    Returns:
        (TurnResult, updated_conversation_history) — the caller
        (Streamlit page) is responsible for persisting the updated
        history back into st.session_state.

    High-level internal flow (implementation deferred):
        1. intent_classifier.classify_intent(...)
        2. branch:
           - NEW_QUERY / FOLLOW_UP -> schema_context_builder ->
             sql_generator.generate_sql (already validated internally,
             see sql_generator.py) -> query_executor.execute_query ->
             result_analyzer.analyze_result ->
             chart_selector.select_chart -> insight_generator (can be
             deferred/async per architecture doc)
           - ML_REQUEST -> ml_plugins.plugin_registry routes to the
             matching plugin
           - CLARIFICATION_NEEDED / OUT_OF_SCOPE -> short-circuit with
             an appropriate response_text, no SQL/ML pipeline invoked
        3. Wrap every step in exception handling that maps internal
           exceptions (exceptions/domain_exceptions.py) to a safe
           TurnResult.error_message — raw exceptions should never
           propagate up to the Streamlit layer uncaught.

    Implementation deferred to implementation phase.
    """
    raise NotImplementedError("Conversation orchestration pending implementation phase.")
