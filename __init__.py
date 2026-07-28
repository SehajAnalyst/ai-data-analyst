"""
AI Data Analyst
================

An AI-powered application that allows users to query relational databases
using plain English. Translates natural language into validated, optimized
SQL, executes it safely, and explains results with charts and business
insights.

Top-level package layout:
    app/            Streamlit presentation layer (UI only, no business logic)
    core/           AI pipeline: orchestration, NL2SQL, execution, insights
    ml_plugins/     Pluggable ML capabilities (future versions)
    db/             Database connectors and internal app persistence
    config/         Centralized, typed configuration management
    utils/          Cross-cutting reusable helpers
    exceptions/     Project-wide exception hierarchy
    logging_setup/  Centralized logging configuration

See docs/ARCHITECTURE.md for the full system design.
"""

__version__ = "0.1.0"
