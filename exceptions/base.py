"""
exceptions/base.py
===================

Root of the project's exception hierarchy.

WHY A CUSTOM EXCEPTION HIERARCHY
----------------------------------
Without this, error handling across the app ends up as a mess of bare
`except Exception` blocks that can't distinguish "the LLM API timed out"
from "the user's SQL was rejected for safety reasons" from "the database
connection failed." Those three situations need three completely
different responses shown to the user, and a flat exception model
makes that impossible to do cleanly.

Every custom exception in this project inherits from AIDataAnalystError.
This lets the Streamlit layer do:

    try:
        ...
    except AIDataAnalystError as e:
        # any error we anticipated — show e.user_message
    except Exception as e:
        # truly unexpected — log full traceback, show generic message

DESIGN RULE: every exception carries both:
  - a `user_message`: safe, friendly, shown directly in the UI
  - the standard exception message: technical detail, goes to logs only

This separation matters because technical error messages (raw SQL,
stack traces, internal table names) should never be shown directly to
end users — that's an information disclosure risk — but they're
essential for debugging and must be logged.
"""

from __future__ import annotations


class AIDataAnalystError(Exception):
    """Base exception for all project-specific errors.

    Args:
        message: Technical message for logs/developers. Can include
            internal details (SQL text, stack info, etc).
        user_message: Safe, human-readable message suitable for direct
            display in the Streamlit UI. Should never leak internal
            schema details, raw SQL, or stack traces.
    """

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or "Something went wrong. Please try again."
