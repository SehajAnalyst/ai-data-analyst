"""
app/main.py
=============

Streamlit entry point. Run with:
    streamlit run app/main.py

This file owns:
  1. Page configuration (must be first Streamlit call).
  2. One-time process startup (logging setup, settings validation).
  3. Session state initialisation on every rerun.
  4. Sidebar rendering.
  5. The main chat interface.

Business logic is never written here. Every heavy operation goes
through app/pipeline.py which calls core/ modules. Every state read
or write goes through app/state/session_state.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page config MUST be the first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap (once per process) ──────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _bootstrap() -> bool:
    """
    Runs once per Streamlit server process (cached by cache_resource).
    Configures logging so every subsequent module gets a working logger.
    Returns True so callers can check it ran.
    """
    from logging_setup.logger import configure_logging
    configure_logging()

    from ml_plugins.bootstrap import register_all_plugins
    register_all_plugins()
    return True


_bootstrap()

# ── Session state init (every rerun) ─────────────────────────────────────

from app.state.session_state import (
    ChatMessage,
    add_chat_message,
    get_chat_history,
    get_conversation_manager,
    init_session_state,
    is_db_connected,
)

init_session_state()

# ── Sidebar ────────────────────────────────────────────────────────────────

from app.components.sidebar import render_sidebar
render_sidebar()

# ── Main content ───────────────────────────────────────────────────────────

from app.components.chat_bubble import render_chat_message
from app.pipeline import run_pipeline

st.title("AI Data Analyst")
st.caption("Ask questions about your database in plain English.")

# ── Render existing chat history ──────────────────────────────────────────
print("Session ID:", st.session_state["session_id"])
print("Chat history length at page load:", len(get_chat_history()))
print("Rendering", len(get_chat_history()), "messages")

for message in get_chat_history():
    print("Rendering:", message.role, "-", message.content)
    render_chat_message(message)
# ── Chat input ─────────────────────────────────────────────────────────────
if not is_db_connected():
    st.info("👈 Connect to a database in the sidebar to get started.")
    st.stop()

user_input = st.chat_input("Ask a question about your data…")

if user_input and user_input.strip():
    question = user_input.strip()

    # Immediately render and persist the user message.
    user_msg = ChatMessage(role="user", content=question)
    add_chat_message(user_msg)
    render_chat_message(user_msg)

    # Build conversation history for follow-up context.
    from core.memory.context_builder import build_conversation_context

    conversation_manager = get_conversation_manager()
    conversation_history = build_conversation_context(
        conversation_manager.get_turns()
    )

    # Run the pipeline with a progress spinner.
    with st.spinner("Thinking…"):
        response = run_pipeline(question, conversation_history)

    # Persist and render the assistant response.
    add_chat_message(response)
    render_chat_message(response)

    # Rerun to update the sidebar query history immediately.
    st.rerun()