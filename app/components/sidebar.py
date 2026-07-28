"""
app/components/sidebar.py
===========================

Renders the entire Streamlit sidebar: database configuration,
connection status, schema summary, query history, and chat controls.

Every interaction that changes state (connecting to a DB, clearing
chat) calls into app/state/session_state.py — this component never
touches st.session_state keys directly.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from app.state.session_state import (
    ChatMessage,
    add_chat_message,
    clear_chat_and_memory,
    clear_chat_history,
    clear_context_only,
    get_conversation_manager,
    get_current_conversation_id,
    get_db_url,
    get_schema,
    invalidate_schema_cache,
    is_db_connected,
    set_current_conversation_id,
    set_db_connected,
    set_db_url,
    get_chat_history,
)
from db.repository.session_factory import get_session
from db.repository.conversation_repository import ConversationRepository
from logging_setup.logger import get_logger

if "confirm_delete" not in st.session_state:
    st.session_state["confirm_delete"] = None

logger = get_logger(__name__)


def render_sidebar() -> None:
    """
    Main entry point called once from app/main.py on every rerun.
    Renders all sidebar sections in order.

    Each section is guarded independently (see _render_section_safely)
    rather than wrapping the whole sidebar in one try/except — added
    during production hardening. One big try/except would mean a
    failure in, say, query history rendering aborts the DB connection
    section too (which the user needs even more urgently if something
    is broken). Independent guarding means a failure in one section
    doesn't take down the others.
    """
    with st.sidebar:
        st.title("AI Data Analyst")
        st.caption("Natural language → SQL → Results")
         
        st.divider()
        _render_section_safely("database", _render_db_section)

        st.divider()
        _render_section_safely("schema", _render_schema_section)

        st.divider()
        _render_section_safely("query_history", _render_query_history)

        st.divider()
        _render_section_safely("controls", _render_controls)


def _render_section_safely(section_name: str, render_fn: Callable[[], None]) -> None:
    """Temporary debugging version."""
    try:
        render_fn()
    except Exception as exc:
        st.exception(exc)   # Shows the real error on the page
        raise               # Shows the full traceback in the terminal too
    
def _render_db_section() -> None:
    """
    Database connection section. Lets the user type a SQLite file path
    (or any SQLAlchemy URL) and connect. Shows current status.

    Connection is validated by trying to create an engine and fetch
    the schema — if either fails, the error is shown inline and
    db_connected is set to False.
    """
    st.subheader("Database")

    current_url = get_db_url() or ""
    new_url = st.text_input(
        "Database URL",
        value=current_url,
        placeholder="sqlite:///./data/sample.db",
        help="SQLAlchemy connection URL. For SQLite: sqlite:///path/to/file.db",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        connect_clicked = st.button(
            "Connect",
            use_container_width=True,
            type="primary",
        )

    with col2:
     if is_db_connected():
        st.markdown("🟢 Connected")
     else:
        st.markdown("🔴 Not Connected")
    if connect_clicked and new_url.strip():
        st.write("DEBUG: Connect button clicked")
        _attempt_connection(new_url.strip())

    elif connect_clicked and not new_url.strip():
        st.warning("Enter a database URL first.")

    if is_db_connected():
        url = get_db_url() or ""

        # Show just the filename for SQLite, full URL for others.
        display = url.split("/")[-1] if "sqlite" in url.lower() else url

        st.caption(f"Connected to: **{display}**")


def _attempt_connection(url: str) -> None:
    st.write("DEBUG: _attempt_connection() was called")

    """
    Tries to connect to the given URL and introspect the schema.
    On success: sets db_url, db_connected, clears cached schema.
    On failure: shows the error, leaves db_connected False.
    """
    from exceptions.domain_exceptions import (
        DatabaseConnectionError,
        DatabaseFileNotFoundError,
        SchemaIntrospectionError,
    )

    with st.spinner("Connecting..."):
        try:
            st.write("STEP 1")

            invalidate_schema_cache()

            st.write("STEP 2")

            set_db_url(url)

            st.write("STEP 3")

            get_schema(url)

            st.write("STEP 4")

            set_db_connected(True)

            st.write("STEP 5")

            st.success("Connected successfully.")
            st.rerun()

        except DatabaseFileNotFoundError as exc:
            set_db_connected(False)
            st.error(exc.user_message)

        except DatabaseConnectionError as exc:
            set_db_connected(False)
            st.error(exc.user_message)

        except SchemaIntrospectionError as exc:
            set_db_connected(False)
            st.error(exc.user_message)

        except Exception as exc:
            set_db_connected(False)
            st.error(f"Connection failed: {exc}")
def _render_schema_section() -> None:
    """
    Renders a compact schema summary (table names and column counts)
    so users know what they can ask about without leaving the chat page.
    Only visible when connected.
    """
    if not is_db_connected():
        st.caption("Connect to a database to see schema.")
        return

    st.subheader("Schema")

    try:
        schema = get_schema()
    except Exception:
        st.warning("Could not load schema.")
        return

    if schema.is_empty:
        st.caption("No tables found in this database.")
        return

    with st.expander(f"{len(schema.tables)} table(s)", expanded=True):
        for table in schema.tables:
            col_count = len(table.columns)
            row_info = f" · ~{table.row_count_estimate:,} rows" if table.row_count_estimate else ""
            st.caption(f"**{table.name}** ({col_count} cols{row_info})")

    if st.button("↺ Refresh Schema", use_container_width=True):
        
        invalidate_schema_cache()
        st.rerun()


def _render_query_history() -> None:
    """
    Shows saved conversations from the internal SQLite database.
    """

    st.subheader("Conversations")
    
    if st.button("➕ New Chat", key="new_chat"):
        clear_chat_history()
        clear_chat_and_memory()
        set_current_conversation_id(None)
        st.rerun()
    
    session = get_session()
    repo = ConversationRepository(session)

    conversations = repo.list_recent_sessions()

    

    if not conversations:
        st.caption("No conversations yet.")
        session.close()
        return

    for conversation in conversations:

        col1, col2 = st.columns([9, 1], vertical_alignment="center")

        with col1:
            clicked = st.button(
                conversation.title,
                key=f"conversation_{conversation.id}",
                width="stretch",
            )

        with col2:
            with st.popover("⋮"):

                rename_clicked = st.button(
                    "✏️ Rename",
                    key=f"rename_{conversation.id}",
                    width="stretch",
                )

                delete_clicked = st.button(
                    "🗑️ Delete",
                    key=f"delete_{conversation.id}",
                    width="stretch",)

                if delete_clicked:
                   st.session_state["confirm_delete"] = conversation.id

        # ---------------- Rename ----------------

        if rename_clicked:
            st.session_state["rename_conversation"] = conversation.id

        if st.session_state.get("rename_conversation") == conversation.id:

            new_title = st.text_input(
                "New conversation name",
                value=conversation.title,
                key=f"title_{conversation.id}",
            )

            if st.button("Save", key=f"save_{conversation.id}"):

                repo.rename_session(
                    conversation.id,
                    new_title,
                )

                del st.session_state["rename_conversation"]
                st.rerun()

        # ---------------- Delete ----------------

        if st.session_state.get("confirm_delete") == conversation.id:
            st.warning("⚠️ Are you sure?")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Yes", key=f"yes_{conversation.id}"):

                    repo.delete_session(conversation.id)

                    if get_current_conversation_id() == conversation.id:
                        clear_chat_and_memory()
                        set_current_conversation_id(None)

                    st.session_state["confirm_delete"] = None
                    session.close()
                    st.rerun()

            with col2:
                if st.button("No", key=f"no_{conversation.id}"):

                    st.session_state["confirm_delete"] = None
                    st.rerun()

        # ---------------- Load Conversation ----------------

        if clicked:

            set_current_conversation_id(conversation.id)

            clear_chat_history()

            turns = repo.get_turns(conversation.id)

            for turn in turns:

                add_chat_message(
                    ChatMessage(
                        role="user",
                        content=turn.user_question,
                    )
                )

                query_result = None

                if turn.generated_sql:
                    from core.execution.query_executor import execute_query
                    from app.state.session_state import (
                        set_db_url,
                        set_db_connected,
                    )

                    try:
                        set_db_url(conversation.database_url)
                        set_db_connected(True)

                        query_result = execute_query(
                            turn.generated_sql,
                            dialect=conversation.db_dialect,
                        )

                    except Exception as exc:
                        print("Failed to reload query:", exc)

                add_chat_message(
                    ChatMessage(
                        role="assistant",
                        content=turn.assistant_response,
                        sql=turn.generated_sql,
                        query_result=query_result,
                    )
                )

            session.close()
            st.rerun()
def _render_controls() -> None:
    """
    Two distinct actions, not one button with two labels:

    "Clear Chat" — full reset. Wipes the visible transcript AND the
    conversation memory used for follow-up resolution. Use this to
    start completely fresh.

    "Clear Context" — wipes ONLY the conversation memory. The visible
    chat transcript stays on screen untouched. Use this to pivot to
    an unrelated question without the AI misapplying old context, but
    without losing what's currently visible.

    These call different session_state functions
    (clear_chat_and_memory vs clear_context_only) because they do
    genuinely different things — see those functions' docstrings in
    app/state/session_state.py.
    """
    st.subheader("Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True, help="Wipe the transcript and conversation memory."):
            clear_chat_and_memory()
            st.rerun()
    with col2:
        if st.button("🧹 Clear Context", use_container_width=True, help="Keep the transcript visible, but stop using it for follow-up context."):
            clear_context_only()
            st.rerun()
