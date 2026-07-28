"""
utils/id_generator.py
========================

Generates session_id and trace_id values (UUIDs) used throughout
logging_setup/, db/models.py, and core/orchestration/ for correlating
a single user turn across logs, audit records, and conversation
history.

Centralized here so ID generation strategy (currently UUID4) can
change in one place if ever needed, rather than `uuid.uuid4()` calls
scattered across the codebase.
"""
