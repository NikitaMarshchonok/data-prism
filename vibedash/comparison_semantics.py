"""Schema-only agreement between two confirmed period definitions."""

from collections.abc import Mapping
from typing import Any

from src.semantic_contract import (
    SemanticContractError,
    semantic_contract_audit_view,
    validate_semantic_contract,
)


def semantic_conflicts(baseline: Mapping, current: Mapping) -> list[dict[str, Any]]:
    """Describe shared columns whose analytical meanings disagree."""
    before = {column["name"]: column for column in baseline["columns"]}
    after = {column["name"]: column for column in current["columns"]}
    return [
        {
            "column": name,
            "baseline_role": before[name]["role"],
            "current_role": after[name]["role"],
            "baseline_unit": before[name]["unit"],
            "current_unit": after[name]["unit"],
        }
        for name in sorted(before.keys() & after.keys())
        if any(before[name][field] != after[name][field] for field in ("role", "unit"))
    ]


def validate_comparison_semantics(baseline_df, current_df, baseline, current):
    """Revalidate both exact schemas and require comparable shared meanings.

    Values are never rescaled. A decimal ratio and a whole percentage must be
    harmonized in the source files before their confirmed units can agree.
    Added/removed columns remain eligible for the schema-change diagnostics.
    """
    before = validate_semantic_contract(baseline_df, baseline)
    after = validate_semantic_contract(current_df, current)
    conflicts = semantic_conflicts(before, after)
    if conflicts:
        column = conflicts[0]["column"]
        raise SemanticContractError(
            f"Shared column '{column}' has different roles or units between periods. "
            "Align the source definitions and scales, then review both contracts. "
            "Values are not automatically converted."
        )
    return before, after


def comparison_semantics_audit(baseline, current):
    return {
        "contract": "confirmed-comparison-semantics-v1",
        "definitions_aligned": True,
        "baseline": semantic_contract_audit_view(baseline),
        "current": semantic_contract_audit_view(current),
    }
