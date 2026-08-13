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
from utils.pdf_export import create_conversation_pdf
import streamlit as st
from pathlib import Path
from core.insights.insight_generator import InsightResult
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
    Database section.

    Users connect to SQLite databases by uploading the database file.
    Previously used database connections are restored automatically
    when a conversation is loaded.
    """

    st.subheader("Database")

    uploaded_db = st.file_uploader(
        "Upload SQLite database",
        type=["db", "sqlite", "sqlite3"],
        help="Upload a SQLite database file.",
    )

    if uploaded_db is not None:
        upload_dir = Path("data")
        upload_dir.mkdir(parents=True, exist_ok=True)

        uploaded_path = upload_dir / uploaded_db.name

        with open(uploaded_path, "wb") as f:
            f.write(uploaded_db.getbuffer())

        database_url = f"sqlite:///{uploaded_path.as_posix()}"

        st.caption(f"📄 {uploaded_db.name}")

        if st.button(
            "Connect Database",
            type="primary",
            use_container_width=True,
        ):
            _attempt_connection(database_url)

    # ── Current connection status ─────────────────────────────────────

    if is_db_connected():
        url = get_db_url() or ""

        if "sqlite" in url.lower():
            display_name = Path(url.replace("sqlite:///", "")).name
        else:
            display_name = url

        st.success(
            f"Connected to **{display_name}**",
            icon="🟢",
        )
    else:
        st.info(
            "No database connected.",
            icon="ℹ️",
        )
def _attempt_connection(url: str) -> None:
    """
    Tries to connect to the given URL and introspect the schema.

    On success:
        - stores the database URL
        - validates the schema
        - marks the database as connected
        - refreshes the UI

    On failure:
        - marks the database as disconnected
        - shows a user-friendly error
    """

    from exceptions.domain_exceptions import (
        DatabaseConnectionError,
        DatabaseFileNotFoundError,
        SchemaIntrospectionError,
    )

    with st.spinner("Connecting..."):
        try:
            # Clear any previously cached schema
            invalidate_schema_cache()

            # Store the new database URL
            set_db_url(url)

            # Validate the connection by loading the schema
            get_schema(url)

            # Connection successful
            set_db_connected(True)

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
    Displays a database overview including:
    - Database dialect
    - Number of tables
    - Total row count
    - Table names
    - Columns and data types
    - Primary/foreign key information
    - Table relationships
    """
    if not is_db_connected():
        st.caption("Connect to a database to see the database overview.")
        return

    st.subheader("Database Overview")

    try:
        schema = get_schema()
    except Exception:
        st.warning("Could not load database schema.")
        return

    if schema.is_empty:
        st.caption("No tables found in this database.")
        return

    # ── Database statistics ─────────────────────────────────────────────

    total_tables = len(schema.tables)

    total_rows = sum(
        table.row_count_estimate or 0
        for table in schema.tables
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Tables", total_tables)

    with col2:
        st.metric("Rows", f"{total_rows:,}")

    st.caption(f"Database type: **{schema.dialect.upper()}**")

    # ── Tables ───────────────────────────────────────────────────────────

    st.markdown("### Tables")

    for table in schema.tables:

        row_info = (
            f"{table.row_count_estimate:,} rows"
            if table.row_count_estimate is not None
            else "row count unavailable"
        )

        with st.expander(
            f"📋 {table.name} · {len(table.columns)} columns · {row_info}"
        ):

            for column in table.columns:

                markers = []

                if column.is_primary_key:
                    markers.append("🔑 PK")

                if column.is_foreign_key:
                    markers.append("🔗 FK")

                if not column.nullable:
                    markers.append("NOT NULL")

                marker_text = (
                    f" · {' · '.join(markers)}"
                    if markers
                    else ""
                )

                st.caption(
                    f"**{column.name}** — "
                    f"`{column.data_type}`"
                    f"{marker_text}"
                )

    # ── Relationships ───────────────────────────────────────────────────

    if schema.relationships:

        st.markdown("### Relationships")

        for relationship in schema.relationships:
            st.caption(
                f"🔗 **{relationship.from_table}.{relationship.from_column}** "
                f"→ "
                f"**{relationship.to_table}.{relationship.to_column}**"
            )

    # ── Refresh ─────────────────────────────────────────────────────────

    if st.button(
        "↺ Refresh Schema",
        use_container_width=True,
    ):
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

        col1, col2 = st.columns(
            [9, 1],
            vertical_alignment="center",
        )

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
                    width="stretch",
                )

                if delete_clicked:
                    st.session_state["confirm_delete"] = conversation.id

        # ---------------------------------------------------------
        # Rename
        # ---------------------------------------------------------

        if rename_clicked:
            st.session_state["rename_conversation"] = conversation.id

        if (
            st.session_state.get("rename_conversation")
            == conversation.id
        ):

            new_title = st.text_input(
                "New conversation name",
                value=conversation.title,
                key=f"title_{conversation.id}",
            )

            if st.button(
                "Save",
                key=f"save_{conversation.id}",
            ):

                repo.rename_session(
                    conversation.id,
                    new_title,
                )

                del st.session_state["rename_conversation"]

                st.rerun()

        # ---------------------------------------------------------
        # Delete
        # ---------------------------------------------------------

        if (
            st.session_state.get("confirm_delete")
            == conversation.id
        ):

            st.warning("⚠️ Are you sure?")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Yes",
                    key=f"yes_{conversation.id}",
                ):

                    repo.delete_session(
                        conversation.id
                    )

                    if (
                        get_current_conversation_id()
                        == conversation.id
                    ):
                        clear_chat_and_memory()
                        set_current_conversation_id(None)

                    st.session_state[
                        "confirm_delete"
                    ] = None

                    session.close()

                    st.rerun()

            with col2:
                if st.button(
                    "No",
                    key=f"no_{conversation.id}",
                ):

                    st.session_state[
                        "confirm_delete"
                    ] = None

                    st.rerun()

        # ---------------------------------------------------------
        # Load Conversation
        # ---------------------------------------------------------

        if clicked:

            set_current_conversation_id(
                conversation.id
            )

            clear_chat_history()

            turns = repo.get_turns(
                conversation.id
            )

            for turn in turns:

                # -------------------------------------------------
                # Restore USER message
                # -------------------------------------------------

                add_chat_message(
                    ChatMessage(
                        role="user",
                        content=turn.user_question,
                    )
                )

                # -------------------------------------------------
                # Restore QUERY RESULT
                # -------------------------------------------------

                query_result = None

                if turn.generated_sql:

                    from core.execution.query_executor import (
                        execute_query,
                    )

                    from app.state.session_state import (
                        set_db_url,
                        set_db_connected,
                    )

                    try:

                        set_db_url(
                            conversation.database_url
                        )

                        set_db_connected(True)

                        query_result = execute_query(
                            turn.generated_sql,
                            dialect=conversation.db_dialect,
                        )

                    except Exception as exc:

                        print(
                            f"Failed to reload query: {exc}"
                        )

                # -------------------------------------------------
                # Restore INSIGHT
                # -------------------------------------------------

                insight = None

                # New conversations:
                # complete insight is stored in chart_metadata.
                if isinstance(
                    turn.insight_data,
                    dict,
                ):

                    from core.insights.insight_generator import (
                        InsightResult,
                    )

                    try:

                        insight = InsightResult(
                            summary=str(
                                turn.insight_data.get(
                                    "summary",
                                    "",
                                )
                            ),

                            key_trends=[
                                str(item)
                                for item in turn.insight_data.get(
                                    "key_trends",
                                    [],
                                )
                            ],

                            outliers=[
                                str(item)
                                for item in turn.insight_data.get(
                                    "outliers",
                                    [],
                                )
                            ],

                            important_metrics=[
                                str(item)
                                for item in turn.insight_data.get(
                                    "important_metrics",
                                    [],
                                )
                            ],

                            follow_up_questions=[
                                str(item)
                                for item in turn.insight_data.get(
                                    "follow_up_questions",
                                    [],
                                )
                            ],

                            is_empty=False,
                        )

                    except Exception as exc:

                        print(
                            f"Failed to restore insight: {exc}"
                        )

                        insight = None

                # -------------------------------------------------
                # FALLBACK FOR OLD CONVERSATIONS
                # -------------------------------------------------
                #
                # Old records were saved before we started storing
                # the complete insight in chart_metadata.
                #
                # They only contain insight_text, so restore the
                # summary instead of showing no insight at all.
                #

                elif turn.insight_text:

                    from core.insights.insight_generator import (
                        InsightResult,
                    )

                    insight = InsightResult(
                        summary=turn.insight_text,
                        is_empty=False,
                    )

                # -------------------------------------------------
                # Restore ASSISTANT message
                # -------------------------------------------------

                add_chat_message(
                    ChatMessage(
                        role="assistant",
                        content=turn.assistant_response,
                        sql=turn.generated_sql,
                        query_result=query_result,
                        insight=insight,
                    )
                )

            session.close()

            st.rerun()
def _render_controls() -> None:
    """
    Chat controls:
    - Clear Chat
    - Clear Context
    - Export Conversation as PDF
    """

    st.subheader("Controls")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🗑 Clear Chat",
            use_container_width=True,
            help="Wipe the transcript and conversation memory.",
        ):
            clear_chat_and_memory()
            st.rerun()

    with col2:
        if st.button(
            "🧹 Clear Context",
            use_container_width=True,
            help="Keep the transcript visible, but stop using it for follow-up context.",
        ):
            clear_context_only()
            st.rerun()

    # ── PDF Export ─────────────────────────────────────────────────────

    st.divider()

    chat_history = get_chat_history()

    if chat_history:

        pdf_bytes = create_conversation_pdf(chat_history)

        st.download_button(
            label="📄 Export Conversation as PDF",
            data=pdf_bytes,
            file_name="ai_data_analyst_conversation.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    else:
        st.button(
            "📄 Export Conversation as PDF",
            disabled=True,
            use_container_width=True,
            help="Start a conversation before exporting it.",
        )