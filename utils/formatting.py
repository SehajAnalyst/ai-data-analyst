"""
utils/formatting.py
=====================

General-purpose display formatting helpers: number formatting
(1234567 -> "1.23M"), date formatting, truncating long strings for
table display, etc. Used by app/components/ (Streamlit UI) and
core/insights/ (when building human-readable result summaries for
insight prompts).

Kept framework-agnostic (no Streamlit imports) so core/ can use these
without pulling in a UI dependency — consistent with the
core-has-zero-Streamlit-dependency rule established throughout core/.
"""
