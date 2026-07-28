"""
core/nl2sql/prompt_builder.py
================================

Loads prompt template text files and performs variable substitution
to produce the final system and user messages sent to the LLM.

WHY TEMPLATE LOADING IS A SEPARATE MODULE FROM sql_generator.py
-----------------------------------------------------------------
sql_generator.py owns the generation *logic* — retry loop, JSON
parsing, confidence derivation, result dataclass assembly. Prompt
string construction is a separate concern: it needs to read files,
format variables, and return strings. If the template loading lived
inside sql_generator.py, prompt iteration (changing wording, adding a
variable, restructuring the schema section) and generation logic
iteration (changing retry count, parsing, confidence heuristics)
would be entangled in the same file. Keeping them separate means:

  - Prompt engineers work in prompt_builder.py and the .txt files,
    not in sql_generator.py.
  - sql_generator.py unit tests can be given pre-built prompt strings
    without depending on the filesystem at all.
  - Template loading can be tested independently with fixed variable
    dicts, without spinning up an LLM.

WHY PYTHON str.format_map INSTEAD OF JINJA2
--------------------------------------------
The templates use {variable} placeholders, which Python's
str.format_map() handles natively — no dependency needed, no
templating engine to maintain, and the format is readable in the
raw .txt file without any special syntax knowledge. Jinja2 would be
warranted if we needed conditional blocks, loops, or filters inside
the templates; we don't in V1. If that changes (e.g. "include this
section only if there are foreign keys"), swap in Jinja2 then — the
interface (load_and_render) stays the same, only the loading
implementation changes.
"""

from __future__ import annotations

from pathlib import Path

from exceptions.domain_exceptions import ConfigurationError
from logging_setup.logger import get_logger

logger = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"
_LLM_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompt_templates"

_template_cache: dict[str, str] = {}


def _load_template(filename: str, directory: Path = _TEMPLATE_DIR) -> str:
    """
    Reads a .txt template file and returns its content. Caches at
    module level for the process lifetime — template files change only
    via deployment, never at runtime (unlike the variables substituted
    into them, which change per request).

    Args:
        filename: the .txt filename (e.g. "sql_generation_system.txt").
        directory: the directory to search. Defaults to
            core/nl2sql/prompt_templates/; use _LLM_TEMPLATE_DIR for
            general-purpose templates in core/llm/prompt_templates/.

    Raises:
        ConfigurationError: the file doesn't exist at the expected path,
            which means a template was renamed or deleted. This is a
            deployment/configuration error, not a runtime error, so
            ConfigurationError is the right type (see
            exceptions/domain_exceptions.py).
    """
    cache_key = str(directory / filename)
    if cache_key in _template_cache:
        return _template_cache[cache_key]

    path = directory / filename
    if not path.is_file():
        raise ConfigurationError(
            message=f"Prompt template not found: {path}",
            user_message="The AI service isn't configured correctly. Contact your administrator.",
        )

    content = path.read_text(encoding="utf-8")
    _template_cache[cache_key] = content
    logger.debug("prompt_template_loaded", filename=filename)
    return content


def build_sql_generation_prompt(
    schema_context: str,
    dialect: str,
    user_question: str,
    conversation_history: str = "",
    retry_context: str = "",
) -> tuple[str, str]:
    """
    Builds the (system_prompt, user_message) pair for SQL generation.

    Args:
        schema_context: formatted schema text from
            schema_context_builder.SchemaContext.formatted_text —
            the actual table/column/relationship information injected
            into the system prompt.
        dialect: database dialect string (e.g. "sqlite") — injected
            into the system prompt so the model knows which SQL
            flavour to use (e.g. SQLite vs PostgreSQL have different
            date functions).
        user_question: the natural-language question.
        conversation_history: compact summary of prior turns, or an
            empty string for a fresh question. Injected into the
            system prompt's conversation history section.
        retry_context: on a regeneration attempt (validator rejected
            the previous SQL), this string contains the rejection
            reason to guide the model's correction. Empty on the
            first attempt. Injected into the user message, NOT the
            system prompt — placing correction feedback in the user
            turn better matches how LLMs handle correction dialogue,
            since modifying the system prompt between calls in a
            session can have unpredictable effects.

    Returns:
        (system_prompt, user_message) — both strings ready to be
        passed to BaseLLMProvider.generate().
    """
    system_template = _load_template("sql_generation_system.txt")
    user_template = _load_template("sql_generation_user.txt")

    system_prompt = system_template.format_map({
        "dialect": dialect,
        "schema_context": schema_context,
        "conversation_history": conversation_history or "(none — this is a new question)",
    })

    formatted_retry = (
        f"NOTE: A previous attempt to generate SQL for this question was rejected.\n"
        f"Rejection reason: {retry_context}\n"
        f"Please correct the issue and regenerate.\n\n"
        if retry_context
        else ""
    )

    user_message = user_template.format_map({
        "user_question": user_question,
        "retry_context": formatted_retry,
    })

    return system_prompt, user_message


def build_sql_explanation_prompt(
    schema_context: str,
    sql_query: str,
    user_question: str,
) -> tuple[str, str]:
    """
    Builds the (system_prompt, user_message) pair for SQL explanation
    (plain-English description of what the SQL does, for non-technical
    users). Used by core/insights/insight_generator.py for the
    "explain this query" part of the results view.

    Returns:
        (system_prompt, user_message).
    """
    template = _load_template("sql_explanation.txt")

    system_prompt = template.format_map({
        "schema_context": schema_context,
        "sql_query": sql_query,
        "user_question": user_question,
    })

    user_message = "Please explain this SQL query in plain English as described."
    return system_prompt, user_message


def build_general_prompt(
    template_filename: str,
    variables: dict[str, str],
    user_message: str,
) -> tuple[str, str]:
    """
    Generic loader for templates in core/llm/prompt_templates/ —
    used for business insights, follow-up questions, data analysis,
    and ML intent detection prompts (the four general-purpose ones not
    specific to the SQL pipeline).

    Args:
        template_filename: filename within core/llm/prompt_templates/.
        variables: substitution dict for the template's {variable}
            placeholders.
        user_message: the user-role message content for this call.

    Returns:
        (system_prompt, user_message).
    """
    template = _load_template(template_filename, directory=_LLM_TEMPLATE_DIR)
    system_prompt = template.format_map(variables)
    return system_prompt, user_message


def clear_cache() -> None:
    """Clears the template file cache. Used in tests that need to
    verify template content changes without a process restart."""
    _template_cache.clear()
