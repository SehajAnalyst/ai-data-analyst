"""
app/pages/1_chat.py
=====================

The primary chat interface — where users type natural-language
questions and see SQL, results, charts, and insights.

RESPONSIBILITIES (UI ONLY)
----------------------------
  - Render the chat input and message history (st.chat_input,
    st.chat_message).
  - Rehydrate conversation_history from st.session_state.
  - On new input, call
    core.orchestration.conversation_manager.handle_turn(
        user_message, conversation_history, session_id
    ) — this is the ONLY core/ call this page makes directly; every
    other concern (SQL generation, validation, execution, charting,
    insight) is internal to that one function.
  - Persist the returned updated conversation_history back into
    st.session_state.
  - Hand off TurnResult fields to app/components/ for rendering:
      - sql_viewer.py renders TurnResult.generated_sql /
        sql_explanation (collapsible — most users want the answer
        first, the SQL on demand, not by default)
      - result_table.py renders TurnResult.query_result
      - chart_renderer.py renders TurnResult.chart_selection against
        TurnResult.query_result
      - chat_bubble.py renders TurnResult.insight_text /
        error_message as the conversational response text

WHY HANDOFF TO COMPONENTS RATHER THAN INLINE RENDERING HERE
-------------------------------------------------------------------
Keeps this page thin and keeps rendering logic for each result type
reusable — app/pages/3_query_history.py will want to re-render a past
turn's SQL/chart/table using the exact same components, not
duplicated rendering code.

Implementation deferred to implementation phase.
"""
