"""Schema-only task context; exact column references, not inferred intent."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


def column_aliases(column: str) -> set[str]:
    original = str(column).strip().lower()
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(column)).lower()
    spaced = re.sub(r"[^a-z0-9а-яё]+", " ", spaced).strip()
    return {alias for alias in {original, spaced, spaced.replace(" ", "_")} if alias}


def prompt_contains_alias(prompt: str, alias: str) -> bool:
    return bool(re.search(
        rf"(?<![a-z0-9а-яё]){re.escape(alias)}(?![a-z0-9а-яё])",
        prompt, flags=re.IGNORECASE,
    ))


def prompt_mentioned_columns(prompt: str, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if any(
        prompt_contains_alias(prompt, alias) for alias in column_aliases(column)
    )]


def prompt_excluded_columns(prompt: str, columns: Iterable[str]) -> set[str]:
    prefix = (
        r"(?:не\s+(?:используй|использовать|учитывай|учитывать)|"
        r"исключи|исключить|игнорируй|игнорировать|"
        r"do\s+not\s+use|don't\s+use|exclude|ignore|without)"
    )
    return {column for column in columns if any(
        re.search(
            rf"{prefix}[^.!?\n]{{0,40}}(?<![a-z0-9а-яё]){re.escape(alias)}(?![a-z0-9а-яё])",
            prompt, flags=re.IGNORECASE,
        ) for alias in column_aliases(column)
    )}


def build_analysis_context(
    prompt: str,
    columns: Iterable[Any],
    profiles: Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prefer explicit measures, then explicit categories, then a confirmed target.

    Does not claim to understand a business task or use an LLM-generated spec
    as evidence of user intent. Exclusions share the dashboard parser's rules.
    """
    names = [str(column) for column in columns]
    by_name = {str(column): profile for column, profile in profiles.items()}
    excluded = prompt_excluded_columns(prompt, names)
    mentioned = [name for name in prompt_mentioned_columns(prompt, names) if name not in excluded]
    measures = [name for name in mentioned if by_name.get(name, {}).get("role") == "measure"]
    groups = [name for name in mentioned if by_name.get(name, {}).get("role") == "category"]
    focus = measures or groups
    source = "explicit_columns" if focus else "unfocused"
    if not focus:
        focus = [name for name in names if name not in excluded and by_name.get(name, {}).get("is_target")]
        if focus:
            source = "confirmed_target"
    return {
        "contract": "analysis-focus-v1",
        "source": source,
        "focus_columns": focus,
        "group_columns": groups,
        "time_columns": [name for name in mentioned if by_name.get(name, {}).get("role") == "temporal"],
        "excluded_columns": [name for name in names if name in excluded],
    }


def finding_columns(finding: Mapping[str, Any]) -> set[str]:
    """Read structured provenance only; titles/narratives are not provenance."""
    columns = finding.get("columns") or []
    result = {str(column) for column in columns} if isinstance(columns, (list, tuple)) else set()
    metrics = finding.get("metrics") or {}
    if isinstance(metrics, Mapping):
        components = metrics.get("component_columns") or []
        if isinstance(components, (list, tuple)):
            result.update(str(column) for column in components)
        for key in ("column", "column_x", "column_y", "value_column"):
            value = metrics.get(key)
            if isinstance(value, str):
                result.add(value)
    return result


def is_relevant(finding: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    columns = finding_columns(finding)
    if columns & set(context.get("excluded_columns") or []):
        return False
    focus = set(context.get("focus_columns") or [])
    return not focus or bool(columns & focus)
