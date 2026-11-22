"""Build the client-facing Data Audit report from one retained analysis.

The report is a deterministic, source-row-free view of results that the
evidence engines already computed.  It never recalculates statistics and never
sends data to an LLM: it selects, orders and labels existing evidence so a
reader without statistical training can act on it, while the appendix keeps
the methodology needed to defend each statement.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


REPORT_CONTRACT = "data-audit-report-v1"
MAX_LABEL_LENGTH = 120
MAX_TEXT_LENGTH = 1200
MAX_PRIORITIES = 3
MAX_QUALITY_ISSUES = 8
MAX_FINDINGS = 6
MAX_TESTS = 8
MAX_EVIDENCE_ITEMS = 4
MAX_NOTES = 8

_READINESS_VERDICTS = {
    "ready": ("good", "Your data is ready for analysis."),
    "ready_with_warnings": (
        "caution",
        "Your data can be analyzed, but some issues should be checked first.",
    ),
    "blocked": (
        "bad",
        "Your data has blocking problems. Fix them before relying on any conclusion.",
    ),
}
_CHECK_STATUS_LABELS = {"blocked": "Must fix", "warning": "Check"}
_CHECK_STATUS_ORDER = {"blocked": 0, "warning": 1}
_CONFIDENCE_LABELS = {
    "high": "Strong evidence",
    "medium": "Moderate evidence",
    "moderate": "Moderate evidence",
    "low": "Weak evidence",
    "exploratory": "Exploratory: verify before acting",
}
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "moderate": 1, "low": 2, "exploratory": 3}
_EXCLUDED_FINDING_CATEGORIES = frozenset({"coverage"})
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# PDF rendering is CPU-bound and runs inside the request thread.  The
# deployment serves requests from a small, fixed thread pool, so concurrent
# renders are capped per process; callers that cannot get a slot quickly
# receive a retryable "busy" response instead of starving every other route.
MAX_CONCURRENT_PDF_RENDERS = 2
PDF_RENDER_WAIT_SECONDS = 5.0
PDF_RENDER_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_PDF_RENDERS)


def clean_text(value: Any, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    """Collapse whitespace and bound user- or engine-provided text."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) > maximum:
        text = text[: maximum - 1].rstrip() + "…"
    return text


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _text_list(values: Any, limit: int = MAX_EVIDENCE_ITEMS) -> list[str]:
    items = [clean_text(item, maximum=400) for item in _sequence(values)]
    return [item for item in items if item][:limit]


def _confidence(value: Any) -> tuple[str, str]:
    key = clean_text(value, maximum=32).lower()
    return key, _CONFIDENCE_LABELS.get(key, "Evidence level not stated")


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _format_p_value(value: Any) -> str:
    if not _finite(value):
        return "n/a"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def _dataset_shape(dashboard: Mapping[str, Any], readiness: Mapping[str, Any]) -> tuple[int | None, int | None]:
    dataset = _mapping(readiness.get("dataset"))
    rows = _int_or_none(dataset.get("row_count"))
    columns = _int_or_none(dataset.get("column_count"))
    shape = dashboard.get("df_shape")
    if (rows is None or columns is None) and isinstance(shape, (list, tuple)) and len(shape) == 2:
        rows = rows if rows is not None else _int_or_none(shape[0])
        columns = columns if columns is not None else _int_or_none(shape[1])
    return rows, columns


def _readiness_section(readiness: Mapping[str, Any]) -> dict[str, Any]:
    status = clean_text(readiness.get("status"), maximum=40)
    tone, verdict = _READINESS_VERDICTS.get(
        status, ("caution", "Data readiness could not be fully assessed.")
    )
    issues = []
    for check in _sequence(readiness.get("checks")):
        check = _mapping(check)
        check_status = clean_text(check.get("status"), maximum=40)
        if check_status not in _CHECK_STATUS_LABELS:
            continue
        issues.append(
            {
                "status": check_status,
                "status_label": _CHECK_STATUS_LABELS[check_status],
                "title": clean_text(check.get("title"), maximum=MAX_LABEL_LENGTH),
                "description": clean_text(check.get("description")),
                "recommendation": clean_text(check.get("recommendation")),
                "evidence": _text_list(check.get("evidence"), 3),
            }
        )
    issues.sort(key=lambda item: _CHECK_STATUS_ORDER[item["status"]])
    counts = _mapping(readiness.get("counts"))
    score = _int_or_none(readiness.get("score"))
    return {
        "status": status or "unknown",
        "tone": tone,
        "verdict": verdict,
        "score": score,
        "summary": clean_text(readiness.get("summary")),
        "passed_count": _int_or_none(counts.get("passed")) or 0,
        "issue_count": len(issues),
        "issues": issues[:MAX_QUALITY_ISSUES],
        "hidden_issue_count": max(0, len(issues) - MAX_QUALITY_ISSUES),
    }


def _priorities_section(brief: Mapping[str, Any]) -> list[dict[str, Any]]:
    priorities = []
    for item in _sequence(brief.get("priorities"))[:MAX_PRIORITIES]:
        item = _mapping(item)
        confidence_key, confidence_label = _confidence(item.get("confidence"))
        priorities.append(
            {
                "title": clean_text(item.get("title"), maximum=MAX_LABEL_LENGTH * 2),
                "finding": clean_text(item.get("finding")),
                "action": clean_text(item.get("action")),
                "risk": clean_text(item.get("risk")),
                "confidence": confidence_key,
                "confidence_label": confidence_label,
            }
        )
    return priorities


def _findings_section(insights: Any) -> list[dict[str, Any]]:
    candidates = []
    for index, item in enumerate(_sequence(insights)):
        item = _mapping(item)
        category = clean_text(item.get("category"), maximum=40).lower()
        if category in _EXCLUDED_FINDING_CATEGORIES:
            continue
        statement = clean_text(item.get("statement"))
        if not statement:
            continue
        confidence_key, confidence_label = _confidence(item.get("confidence"))
        severity = clean_text(item.get("severity"), maximum=20).lower()
        candidates.append(
            (
                _SEVERITY_ORDER.get(severity, 3),
                _CONFIDENCE_ORDER.get(confidence_key, 4),
                index,
                {
                    "title": clean_text(item.get("title"), maximum=MAX_LABEL_LENGTH),
                    "category": category or "finding",
                    "statement": statement,
                    "recommendation": clean_text(item.get("recommendation")),
                    "evidence": _text_list(item.get("evidence")),
                    "sample_size": _int_or_none(item.get("sample_size")),
                    "severity": severity or "info",
                    "confidence": confidence_key,
                    "confidence_label": confidence_label,
                },
            )
        )
    candidates.sort(key=lambda entry: entry[:3])
    return [entry[3] for entry in candidates[:MAX_FINDINGS]]


def _statistics_section(validation: Mapping[str, Any]) -> dict[str, Any]:
    tests = []
    for test in _sequence(validation.get("tests")):
        test = _mapping(test)
        effect = _mapping(test.get("effect_size"))
        interval = _mapping(test.get("confidence_interval"))
        effect_value = effect.get("value")
        lower, upper = interval.get("lower"), interval.get("upper")
        tests.append(
            {
                "title": clean_text(test.get("title"), maximum=MAX_LABEL_LENGTH * 2),
                "method": clean_text(test.get("method"), maximum=MAX_LABEL_LENGTH),
                "sample_size": _int_or_none(test.get("sample_size")),
                "adjusted_p_value": _format_p_value(test.get("adjusted_p_value")),
                "significant": test.get("significant") is True,
                "effect": (
                    f"{clean_text(effect.get('name'), maximum=40)} = {effect_value:.3f}"
                    f" ({clean_text(effect.get('magnitude'), maximum=20)})"
                    if _finite(effect_value)
                    else "n/a"
                ),
                "interval": (
                    f"[{lower:.3f}, {upper:.3f}]"
                    if _finite(lower) and _finite(upper)
                    else "n/a"
                ),
                "interpretation": clean_text(test.get("interpretation")),
            }
        )
    status = clean_text(validation.get("status"), maximum=40) or "unavailable"
    return {
        "status": status,
        "withheld": status != "ok",
        "summary": clean_text(validation.get("summary")),
        "correction": clean_text(validation.get("correction"), maximum=MAX_LABEL_LENGTH),
        "alpha": validation.get("alpha") if _finite(validation.get("alpha")) else None,
        "candidate_test_count": _int_or_none(validation.get("candidate_test_count")) or 0,
        "significant_test_count": _int_or_none(validation.get("significant_test_count")),
        "tests": tests[:MAX_TESTS],
        "methodology_notes": _text_list(validation.get("methodology_notes"), MAX_NOTES),
    }


def build_audit_report_context(
    session_data: Mapping[str, Any],
    *,
    prepared_for: str = "",
    prepared_by: str = "",
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Return the deterministic view model for the Data Audit report."""
    if not isinstance(session_data, Mapping):
        raise ValueError("The analysis result is invalid.")
    dashboard = session_data.get("dashboard_data")
    if not isinstance(dashboard, Mapping):
        raise ValueError("The analysis result has no evidence to report.")

    readiness = _mapping(dashboard.get("readiness"))
    brief = _mapping(dashboard.get("decision_brief"))
    manifest = _mapping(session_data.get("audit_manifest"))
    dataset_manifest = _mapping(manifest.get("dataset"))
    rows, columns = _dataset_shape(dashboard, readiness)
    moment = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)

    return {
        "contract": REPORT_CONTRACT,
        "title": "Data Audit Report",
        "prepared_for": clean_text(prepared_for, maximum=MAX_LABEL_LENGTH),
        "prepared_by": clean_text(prepared_by, maximum=MAX_LABEL_LENGTH),
        # Month names are spelled explicitly: strftime("%B") follows the
        # server locale and would make the same report differ by host.
        "generated_on": f"{moment.day} {_MONTH_NAMES[moment.month - 1]} {moment.year}",
        "generated_at": moment.strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": {
            "name": clean_text(session_data.get("filename"), maximum=255) or "Uploaded dataset",
            "rows": rows,
            "columns": columns,
            "question": clean_text(session_data.get("prompt"), maximum=600),
        },
        "summary": {
            "headline": clean_text(brief.get("headline"), maximum=300),
            "text": clean_text(brief.get("summary")),
            "guardrail": clean_text(brief.get("guardrail")),
        },
        "priorities": _priorities_section(brief),
        "readiness": _readiness_section(readiness),
        "findings": _findings_section(dashboard.get("insights")),
        "statistics": _statistics_section(_mapping(dashboard.get("statistical_validation"))),
        "audit": {
            "analysis_contract": clean_text(manifest.get("analysis_contract"), maximum=80),
            "engine_version": clean_text(manifest.get("engine_version"), maximum=128),
            "content_sha256": clean_text(dataset_manifest.get("content_sha256"), maximum=64),
            "analyzed_rows": _int_or_none(dataset_manifest.get("analyzed_rows")),
            "truncated": dataset_manifest.get("truncated") is True,
        },
    }
