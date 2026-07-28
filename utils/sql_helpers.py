"""
utils/sql_helpers.py
======================

Small, stateless SQL string/AST utility functions shared across
core/nl2sql/ and core/execution/ — things like normalizing whitespace
for comparison, extracting table names from a parsed AST for logging,
or pretty-formatting SQL for display in the UI.

WHY THIS IS IN utils/ AND NOT INSIDE sql_validator.py OR sql_generator.py
-------------------------------------------------------------------------------
These are genuinely cross-cutting helpers with no business logic and
no safety implications of their own — pretty-printing SQL for display
is not a safety-relevant operation, unlike everything in
sql_validator.py. Keeping them separate avoids bloating the
safety-critical validator file with formatting concerns, and avoids
sql_generator.py and sql_validator.py each reinventing the same
string-handling utilities independently.

Implementation deferred — this file currently exists to establish
where these helpers belong.
"""
