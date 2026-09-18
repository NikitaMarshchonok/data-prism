"""Small, deterministic view helpers for period-comparison reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_MAX_COUNT = 100_000
# A tuple deliberately avoids hashing an untrusted ``str`` subclass during the
# membership check below.  JSON normally gives us plain strings, but retained
# payloads should remain deterministic even if a caller supplies an odd value.
_SAFE_STATUSES = ("stable", "review_required", "unknown")


def _count(value: Any) -> int:
    """Return a bounded non-negative count without exposing input details."""
    if isinstance(value, bool):
        return 0
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(_MAX_COUNT, value))


def _block(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_comparison_decision_guidance(report: Any) -> dict[str, Any]:
    """Build bounded, aggregate-only checks for the comparison view.

    This intentionally does not copy labels, column names, filenames, rows, or
    free-form text from the result.  It is guidance for review, not a causal
    recommendation, and remains deterministic for a given aggregate report.
    """
    comparison = _block(report)
    status = comparison.get("status")
    # JSON-shaped reports can still be malformed (for example, an array or
    # object in place of the scalar status).  Guard the set lookup so guidance
    # remains deterministic and aggregate-only for every input shape.
    if not isinstance(status, str) or status not in _SAFE_STATUSES:
        status = "unknown"
    inputs = _block(comparison.get("input"))
    baseline = _block(inputs.get("baseline"))
    current = _block(inputs.get("current"))
    schema = _block(comparison.get("schema_changes"))
    metrics = _block(comparison.get("numeric_metrics"))
    drift = _block(comparison.get("distribution_drift"))
    schema_change_count = sum(
        _count(len(schema.get(key) or []) if isinstance(schema.get(key), list) else 0)
        for key in ("new_columns", "missing_columns", "type_changes")
    )
    snapshot = {
        "status": status,
        "baseline_rows": _count(baseline.get("rows")),
        "current_rows": _count(current.get("rows")),
        "eligible_metrics": _count(metrics.get("eligible_count")),
        "tested_metrics": _count(metrics.get("tested_count")),
        "significant_metrics": _count(metrics.get("significant_count")),
        "drift_signals": _count(len(drift.get("signals") or []) if isinstance(drift.get("signals"), list) else 0),
        "schema_changes": min(_MAX_COUNT, schema_change_count),
    }
    questions = [
        "Confirm the row unit, metric definitions, and population are aligned across both periods.",
        "Review the displayed aggregate differences against data-quality and coverage checks before taking action.",
    ]
    if snapshot["schema_changes"]:
        questions.append("Check schema changes and type differences before comparing metric values.")
    if snapshot["drift_signals"]:
        questions.append("Inspect the distribution-drift signals and missingness context for the affected features.")
    if snapshot["tested_metrics"] and snapshot["significant_metrics"]:
        questions.append("Check whether flagged statistical differences are relevant to the decision context, without treating significance as a decision rule.")
    if not snapshot["tested_metrics"]:
        questions.append("Confirm there is sufficient finite numeric evidence for any metric comparison you plan to use.")
    return {
        "contract": "comparison-decision-guidance-v1",
        "evidence_snapshot": snapshot,
        "questions": questions[:5],
    }
