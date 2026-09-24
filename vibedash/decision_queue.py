"""Deterministic review-state grouping for evidence-linked decisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any


DECISION_QUEUE_WINDOW_DAYS = 7
ALLOWED_DECISION_QUEUE_VIEWS = frozenset(
    {"active", "overdue", "today", "upcoming", "closed", "all"}
)
ALLOWED_DECISION_STATUSES = frozenset(
    {"tracking", "validated", "invalidated", "cancelled"}
)


def _review_state(decision_case: Mapping[str, Any], today: date) -> dict[str, Any]:
    try:
        review_date = date.fromisoformat(str(decision_case.get("review_date") or ""))
    except ValueError as error:
        raise ValueError("Decision case review date is invalid.") from error

    status = decision_case.get("status")
    if status not in ALLOWED_DECISION_STATUSES:
        raise ValueError("Decision case status is invalid.")
    days_until_review = (review_date - today).days
    if status != "tracking":
        state = "closed"
        timing_label = "Closed"
    elif days_until_review < 0:
        state = "overdue"
        count = abs(days_until_review)
        timing_label = f"Overdue by {count} day" + ("" if count == 1 else "s")
    elif days_until_review == 0:
        state = "today"
        timing_label = "Due today"
    elif days_until_review <= DECISION_QUEUE_WINDOW_DAYS:
        state = "upcoming"
        timing_label = f"Due in {days_until_review} day" + (
            "" if days_until_review == 1 else "s"
        )
    else:
        state = "scheduled"
        timing_label = f"Scheduled for {review_date.isoformat()}"

    return {
        **decision_case,
        "review_state": state,
        "days_until_review": days_until_review,
        "timing_label": timing_label,
    }


def build_decision_queue(
    decision_cases: Iterable[Mapping[str, Any]],
    *,
    view: str = "active",
    today: date | None = None,
) -> dict[str, Any]:
    """Return scoped decision cases grouped and ordered by review urgency."""
    if view not in ALLOWED_DECISION_QUEUE_VIEWS:
        raise ValueError("Unsupported decision queue view.")
    effective_today = today or date.today()
    if type(effective_today) is not date:
        raise ValueError("today must be a date.")

    cases = list(decision_cases)
    if any(not isinstance(decision_case, Mapping) for decision_case in cases):
        raise ValueError("Decision cases must be objects.")
    enriched = [
        _review_state(decision_case, effective_today)
        for decision_case in cases
    ]
    active = sorted(
        (item for item in enriched if item["review_state"] != "closed"),
        key=lambda item: (
            item["review_date"],
            item.get("created_at") or "",
            item.get("id") or "",
        ),
    )
    closed = sorted(
        (item for item in enriched if item["review_state"] == "closed"),
        key=lambda item: (
            item.get("updated_at") or "",
            item.get("id") or "",
        ),
        reverse=True,
    )
    ordered = [*active, *closed]
    filters = {
        "active": lambda item: item["review_state"] != "closed",
        "overdue": lambda item: item["review_state"] == "overdue",
        "today": lambda item: item["review_state"] == "today",
        "upcoming": lambda item: item["review_state"] == "upcoming",
        "closed": lambda item: item["review_state"] == "closed",
        "all": lambda _item: True,
    }
    counts = {
        "active": len(active),
        "overdue": sum(item["review_state"] == "overdue" for item in enriched),
        "today": sum(item["review_state"] == "today" for item in enriched),
        "upcoming": sum(item["review_state"] == "upcoming" for item in enriched),
        "closed": len(closed),
        "all": len(enriched),
    }
    return {
        "view": view,
        "today": effective_today.isoformat(),
        "window_days": DECISION_QUEUE_WINDOW_DAYS,
        "counts": counts,
        "items": [item for item in ordered if filters[view](item)],
    }
