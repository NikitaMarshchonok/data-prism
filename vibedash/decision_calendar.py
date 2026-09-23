"""Build privacy-preserving calendar reminders for decision reviews."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping


def _calendar_text(value: Any) -> str:
    """Return RFC 5545 TEXT with control characters and separators escaped."""
    normalized = " ".join(str(value or "").split())
    return (
        normalized.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
    )


def _fold_line(line: str) -> list[str]:
    """Fold one content line at 75 UTF-8 octets without splitting characters."""
    remaining = line.encode("utf-8")
    folded: list[str] = []
    first = True
    while remaining:
        # Continuation lines begin with one space, leaving 74 content octets.
        limit = 75 if first else 74
        end = min(limit, len(remaining))
        while end < len(remaining) and remaining[end] & 0xC0 == 0x80:
            end -= 1
        chunk = remaining[:end].decode("utf-8")
        folded.append(("" if first else " ") + chunk)
        remaining = remaining[end:]
        first = False
    return folded or [""]


def _created_timestamp(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError("Decision case creation time is invalid.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_decision_review_calendar(decision_case: Mapping[str, Any]) -> bytes:
    """Serialize one bounded decision case as an all-day review reminder."""
    if not isinstance(decision_case, Mapping):
        raise ValueError("Decision case is invalid.")

    case_id = str(decision_case.get("id") or "")
    if len(case_id) != 32 or any(character not in "0123456789abcdef" for character in case_id):
        raise ValueError("Decision case identifier is invalid.")
    try:
        review_date = date.fromisoformat(str(decision_case.get("review_date") or ""))
    except ValueError as error:
        raise ValueError("Decision case review date is invalid.") from error
    try:
        review_end = review_date + timedelta(days=1)
    except OverflowError as error:
        raise ValueError("Decision case review date is outside the calendar range.") from error

    snapshot = decision_case.get("evidence_snapshot")
    priority = snapshot.get("priority") if isinstance(snapshot, Mapping) else None
    title = priority.get("title") if isinstance(priority, Mapping) else None
    summary_subject = title or decision_case.get("decision") or "Decision outcome"
    description_parts = (
        ("Decision", decision_case.get("decision")),
        ("Owner", decision_case.get("owner")),
        ("Success metric", decision_case.get("success_metric")),
        ("Target outcome", decision_case.get("target_outcome")),
    )
    description = "\\n".join(
        f"{label}: {_calendar_text(value)}"
        for label, value in description_parts
        if str(value or "").strip()
    )

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Data Prism//Decision Review//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{case_id}@data-prism.local",
        f"DTSTAMP:{_created_timestamp(decision_case.get('created_at'))}",
        f"DTSTART;VALUE=DATE:{review_date.strftime('%Y%m%d')}",
        f"DTEND;VALUE=DATE:{review_end.strftime('%Y%m%d')}",
        f"SUMMARY:{_calendar_text(f'Review decision: {summary_subject}')}",
        f"DESCRIPTION:{description}",
        "TRANSP:TRANSPARENT",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    folded = [part for line in lines for part in _fold_line(line)]
    return ("\r\n".join(folded) + "\r\n").encode("utf-8")
