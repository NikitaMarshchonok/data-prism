"""Build bounded view data for standalone decision-case reports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any


_IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_STATUSES = ("tracking", "validated", "invalidated", "cancelled")
_STATUS_LABELS = {
    "tracking": "Tracking",
    "validated": "Target validated",
    "invalidated": "Target invalidated",
    "cancelled": "Decision cancelled",
}


def _text(
    value: Any,
    label: str,
    *,
    maximum: int,
    required: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid.")
    normalized = " ".join(value.split())
    if required and not normalized:
        raise ValueError(f"{label} is unavailable.")
    if len(normalized) > maximum:
        raise ValueError(f"{label} exceeds the report limit.")
    return normalized


def _timestamp(value: Any, label: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} is invalid.") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone.")
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def build_decision_report_context(decision_case: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic, source-row-free decision report contract."""
    if not isinstance(decision_case, Mapping):
        raise ValueError("Decision case is invalid.")
    case_id = decision_case.get("id")
    if not isinstance(case_id, str) or not _IDENTIFIER_PATTERN.fullmatch(case_id):
        raise ValueError("Decision case identifier is invalid.")
    status = decision_case.get("status")
    if not isinstance(status, str) or status not in _SAFE_STATUSES:
        raise ValueError("Decision case status is invalid.")
    priority_number = decision_case.get("priority")
    if (
        isinstance(priority_number, bool)
        or not isinstance(priority_number, int)
        or not 1 <= priority_number <= 3
    ):
        raise ValueError("Decision priority is invalid.")
    try:
        review_date = date.fromisoformat(
            _text(decision_case.get("review_date"), "Review date", maximum=10)
        )
    except ValueError as error:
        raise ValueError("Review date is invalid.") from error

    snapshot = decision_case.get("evidence_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Evidence snapshot is invalid.")
    if snapshot.get("contract") != "decision-case-source-v1":
        raise ValueError("Evidence snapshot contract is unsupported.")
    priority = snapshot.get("priority")
    if not isinstance(priority, Mapping):
        raise ValueError("Evidence priority is invalid.")
    evidence = priority.get("evidence") or []
    if not isinstance(evidence, list) or any(
        not isinstance(item, str) for item in evidence
    ):
        raise ValueError("Evidence items are invalid.")
    evidence_items = [
        _text(item, "Evidence item", maximum=300)
        for item in evidence[:5]
    ]

    fingerprint = snapshot.get("dataset_sha256")
    if not isinstance(fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
        fingerprint
    ):
        fingerprint = "Unavailable"

    analysis_contract = snapshot.get("analysis_contract")
    if analysis_contract in (None, ""):
        analysis_contract = "Unavailable"
    confidence = priority.get("confidence")
    if confidence in (None, ""):
        confidence = "unknown"

    actual_outcome = _text(
        decision_case.get("actual_outcome", ""),
        "Actual outcome",
        maximum=1200,
        required=False,
    )
    resolved_at = _timestamp(
        decision_case.get("resolved_at"),
        "Resolution time",
        required=False,
    )
    return {
        "contract": "decision-case-report-v1",
        "case_id": case_id,
        "short_id": case_id[:12],
        "priority_number": priority_number,
        "status": status,
        "status_label": _STATUS_LABELS[status],
        "owner": _text(decision_case.get("owner"), "Owner", maximum=120),
        "decision": _text(
            decision_case.get("decision"), "Decision", maximum=1000
        ),
        "success_metric": _text(
            decision_case.get("success_metric"),
            "Success metric",
            maximum=240,
        ),
        "target_outcome": _text(
            decision_case.get("target_outcome"),
            "Target outcome",
            maximum=500,
        ),
        "review_date": review_date.isoformat(),
        "actual_outcome": actual_outcome,
        "outcome_recorded": bool(actual_outcome),
        "created_at": _timestamp(decision_case.get("created_at"), "Creation time"),
        "updated_at": _timestamp(decision_case.get("updated_at"), "Update time"),
        "resolved_at": resolved_at,
        "evidence": {
            "title": _text(
                priority.get("title"), "Evidence title", maximum=200
            ),
            "finding": _text(
                priority.get("finding", ""),
                "Evidence finding",
                maximum=800,
                required=False,
            ),
            "recommended_action": _text(
                priority.get("recommended_action", ""),
                "Recommended action",
                maximum=800,
                required=False,
            ),
            "confidence": _text(
                confidence,
                "Evidence confidence",
                maximum=40,
            ),
            "items": evidence_items,
            "analysis_contract": _text(
                analysis_contract,
                "Analysis contract",
                maximum=80,
            ),
            "dataset_sha256": fingerprint,
        },
    }
