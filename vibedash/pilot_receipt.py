"""Privacy-safe, portable evidence receipt for one VibeDash scope.

The receipt deliberately contains aggregate counters only.  It is intended to
survive an ephemeral demo deployment without turning account, job, decision,
or browser identifiers into an export surface.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .pilot_metrics import NEXT_CYCLE_INTENT, PERCEIVED_TIME_SAVED


PILOT_RECEIPT_CONTRACT = "pilot-evidence-receipt-v1"
MAX_PILOT_RECEIPT_BYTES = 64 * 1024

_OUTCOME_CONTRACT = "decision-outcome-summary-v1"
_VALUE_CONTRACT = "pilot-scope-value-v1"
_VERSION_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,64}$")
_COUNT_FIELDS = (
    "retained_cases",
    "tracking_cases",
    "closed_cases",
    "evaluated_cases",
    "validated_cases",
    "invalidated_cases",
    "cancelled_cases",
)
_RATIO_FIELDS = (
    "resolution_rate",
    "evaluation_rate",
    "validation_share_among_evaluated",
)


class PilotReceiptError(ValueError):
    """The aggregate receipt contract or serialized size is invalid."""


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotReceiptError("Pilot receipt count is invalid.")
    return value


def _ratio(value: Any) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
    ):
        raise PilotReceiptError("Pilot receipt ratio is invalid.")
    return round(float(value), 4)


def _timestamp(value: datetime | None) -> str:
    value = value or datetime.now(timezone.utc)
    if not isinstance(value, datetime):
        raise PilotReceiptError("Pilot receipt timestamp is invalid.")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _outcome_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("contract") != _OUTCOME_CONTRACT:
        raise PilotReceiptError("Decision outcome summary is invalid.")
    result: dict[str, Any] = {"contract": _OUTCOME_CONTRACT}
    for field in _COUNT_FIELDS:
        result[field] = _count(value.get(field))
    for field in _RATIO_FIELDS:
        result[field] = _ratio(value.get(field))
    return result


def _fixed_choice_counts(value: Any, choices: tuple[str, ...]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise PilotReceiptError("Pilot value feedback counts are invalid.")
    return {choice: _count(value.get(choice)) for choice in choices}


def _value_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("contract") != _VALUE_CONTRACT:
        raise PilotReceiptError("Pilot value feedback summary is invalid.")
    cohort_days = _count(value.get("cohort_days"))
    if not 1 <= cohort_days <= 30:
        raise PilotReceiptError("Pilot value feedback window is invalid.")
    installed = value.get("collection_installed")
    if not isinstance(installed, bool):
        raise PilotReceiptError("Pilot measurement installation state is invalid.")
    return {
        "contract": _VALUE_CONTRACT,
        "cohort_days": cohort_days,
        "collection_installed": installed,
        "completed_opted_in_analyses": _count(
            value.get("completed_opted_in_analyses")
        ),
        "feedback_responses": _count(value.get("feedback_responses")),
        "value_feedback_responses": _count(
            value.get("value_feedback_responses")
        ),
        "value_feedback_response_rate_among_completed": _ratio(
            value.get("value_feedback_response_rate_among_completed")
        ),
        "legacy_feedback_responses_without_value_signals": _count(
            value.get("legacy_feedback_responses_without_value_signals")
        ),
        "perceived_time_saved": _fixed_choice_counts(
            value.get("perceived_time_saved"), PERCEIVED_TIME_SAVED
        ),
        "next_cycle_intent": _fixed_choice_counts(
            value.get("next_cycle_intent"), NEXT_CYCLE_INTENT
        ),
    }


def build_pilot_evidence_receipt(
    outcome_summary: Mapping[str, Any],
    value_summary: Mapping[str, Any],
    *,
    service_version: str,
    decision_retention_days: int,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an allowlisted aggregate receipt for one already scoped reader."""
    if (
        not isinstance(service_version, str)
        or service_version != service_version.strip()
        or not _VERSION_RE.fullmatch(service_version)
    ):
        raise PilotReceiptError("Service version is invalid.")
    retention_days = _count(decision_retention_days)
    if not 1 <= retention_days <= 3650:
        raise PilotReceiptError("Decision retention window is invalid.")
    safe_value_summary = _value_summary(value_summary)
    return {
        "contract": PILOT_RECEIPT_CONTRACT,
        "generated_at": _timestamp(generated_at),
        "service_version": service_version,
        "scope_boundary": "current-signed-vibedash-scope",
        "windows": {
            "decision_retention_days": retention_days,
            "pilot_feedback_days": safe_value_summary["cohort_days"],
        },
        "decision_outcomes": _outcome_summary(outcome_summary),
        "value_feedback": safe_value_summary,
        "limitations": [
            "This receipt contains retained aggregate state for one signed VibeDash scope, not a verified person or company.",
            "Decision outcomes and time-saved values are user-entered or self-reported and do not establish causality or independently verified savings.",
            "Next-cycle intent is not observed repeat use, adoption, or willingness to pay.",
            "Missing, expired, withdrawn, or redeployed state can reduce the reported counts.",
        ],
    }


def serialize_pilot_evidence_receipt(document: Mapping[str, Any]) -> bytes:
    """Serialize a receipt with a small, deterministic byte budget."""
    if not isinstance(document, Mapping) or document.get("contract") != PILOT_RECEIPT_CONTRACT:
        raise PilotReceiptError("Pilot receipt document is invalid.")
    try:
        payload = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PilotReceiptError("Pilot receipt could not be serialized.") from error
    if len(payload) > MAX_PILOT_RECEIPT_BYTES:
        raise PilotReceiptError("Pilot receipt exceeds its size limit.")
    return payload
