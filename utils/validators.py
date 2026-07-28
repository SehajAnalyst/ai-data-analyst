"""
utils/validators.py
=====================

Generic input validation helpers NOT specific to SQL safety (that's
core/nl2sql/sql_validator.py's job, and should stay there given how
safety-critical it is — see that file's docstring on why it shouldn't
be diluted with unrelated concerns).

This file is for things like: validating a user-provided database
connection string is well-formed before attempting to connect,
validating a session_id is a well-formed UUID, etc. — ordinary input
hygiene, not safety enforcement.
"""
