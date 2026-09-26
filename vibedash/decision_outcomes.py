"""Bounded aggregate outcomes for one decision-case scope."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_STATUSES = ("tracking", "validated", "invalidated", "cancelled")


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def build_decision_outcome_summary(
    status_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a display contract from already scope-filtered status counts."""
    if not isinstance(status_counts, Mapping):
        raise ValueError("Decision outcome counts are invalid.")
    if any(status not in _STATUSES for status in status_counts):
        raise ValueError("Decision outcome status is invalid.")
    counts = {}
    for status in _STATUSES:
        count = status_counts.get(status, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Decision outcome count is invalid.")
        counts[status] = count

    total = sum(counts.values())
    evaluated = counts["validated"] + counts["invalidated"]
    closed = evaluated + counts["cancelled"]
    return {
        "contract": "decision-outcome-summary-v1",
        "retained_cases": total,
        "tracking_cases": counts["tracking"],
        "closed_cases": closed,
        "evaluated_cases": evaluated,
        "validated_cases": counts["validated"],
        "invalidated_cases": counts["invalidated"],
        "cancelled_cases": counts["cancelled"],
        "resolution_rate": _ratio(closed, total),
        "evaluation_rate": _ratio(evaluated, total),
        "validation_share_among_evaluated": _ratio(
            counts["validated"], evaluated
        ),
    }
