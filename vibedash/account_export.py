"""Pure, privacy-preserving account export builder.

The functions in this module intentionally accept already loaded values.  They
do not know about the account, job, decision-case, session, or filesystem
stores.  Keeping this boundary pure makes it possible for a route to enforce
ownership before calling the builder and makes the export contract easy to
test in isolation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any


ACCOUNT_EXPORT_CONTRACT = "vibedash-account-export-v1"
MAX_ACCOUNT_EXPORT_JOBS = 50
MAX_ACCOUNT_EXPORT_CASES = 100
MAX_ACCOUNT_EXPORT_BYTES = 2 * 1024 * 1024


class AccountExportError(ValueError):
    """A safe, deterministic account-export validation or size failure."""


class AccountExportTooLargeError(AccountExportError):
    """The serialized export is larger than its byte budget."""


# Compatibility spelling used by the route's status mapper and by callers
# that prefer a size-oriented exception name.
AccountExportSizeError = AccountExportTooLargeError


_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_TIMESTAMP_MAX = 64
_TEXT_MAX = 512
_STATUS_MAX = 64
_CONTRACT_MAX = 128
_ERROR_CODE_MAX = 64
_EMAIL_MAX = 254

_JOB_STATUSES = frozenset({"queued", "running", "completed", "failed"})
_DECISION_STATUSES = frozenset({"tracking", "validated", "invalidated", "cancelled"})
_ANALYSIS_KINDS = frozenset({"period_comparison"})
_ERROR_CODES = frozenset({
    "analysis_failed",
    "dispatch_failed",
    "invalid_dataset",
    "worker_interrupted",
})
_ANALYSIS_CONTRACTS = frozenset({
    "vibedash-evidence-v2",
    "vibedash-period-comparison-v1",
})
_READINESS_CONTRACTS = frozenset({"dataset-readiness-v1"})
_DECISION_BRIEF_CONTRACTS = frozenset({"decision-brief-v1"})
_DECISION_CASE_CONTRACTS = frozenset({"decision-case-source-v1"})
_READINESS_STATUSES = frozenset({"blocked", "ready_with_warnings", "ready"})
_DECISION_BRIEF_STATUSES = frozenset({"blocked", "review_required", "insufficient_evidence"})
_COMPARISON_STATUSES = frozenset({"stable", "review_required"})
_COMPARISON_CONTRACTS = frozenset({"period-comparison-v1"})


def _text(value: Any, maximum: int, *, allow_empty: bool = False) -> str | None:
    """Return a bounded string, or ``None`` for malformed values."""
    if not isinstance(value, str):
        return None
    # Reject non-UTF-8 surrogate data before it reaches the byte serializer.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    value = " ".join(value.split())
    if not allow_empty and not value:
        return None
    return value[:maximum]


def _timestamp(value: Any) -> str | None:
    """Keep only an ISO-like timestamp with a conservative length bound."""
    value = _text(value, _TIMESTAMP_MAX)
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        # Stored account/job timestamps are expected to be UTC.  Normalizing
        # naive input is safer than preserving an ambiguous timezone.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _review_date(value: Any) -> str | None:
    """Keep only the decision-case store's canonical calendar-date format."""
    value = _text(value, 10)
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        return None


def _generated_timestamp(value: Any) -> str | None:
    """Normalize an explicit datetime or timestamp string for the envelope."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    return _timestamp(value)


def _safe_int(value: Any, *, minimum: int = 0, maximum: int = 2**53 - 1) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not minimum <= value <= maximum:
        return None
    return value


def _safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _safe_id(value: Any) -> str | None:
    value = _text(value, 128)
    if value is None or _ID_RE.fullmatch(value) is None:
        return None
    return value


def _safe_hash(value: Any) -> str | None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        return None
    return value.lower()


def _safe_contract(value: Any, *, expected: str | None = None) -> str | None:
    value = _text(value, _CONTRACT_MAX)
    if value is None:
        return None
    if expected is not None and value != expected:
        return None
    # Contracts are application constants; accepting arbitrary user text here
    # would provide a side channel for data that the allowlist excludes.
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
        return None
    return value


def _enum(value: Any, allowed: frozenset[str]) -> str | None:
    """Keep only one of the contract's closed string enums."""
    return value if isinstance(value, str) and value in allowed else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_count(value: Any) -> int | None:
    return _safe_int(value, minimum=0, maximum=2**31 - 1)


def _safe_score(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or value > 1:
        return None
    # Keep integer scores as integers and avoid serializing accidental float
    # representations with platform-specific precision.
    return int(value) if isinstance(value, int) else round(float(value), 6)


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _normal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _mapping(manifest.get("dataset"))
    specification = _mapping(manifest.get("specification"))
    evidence = _mapping(manifest.get("evidence"))
    readiness = _mapping(evidence.get("readiness"))
    decision_brief = _mapping(evidence.get("decision_brief"))
    result: dict[str, Any] = {"kind": "normal"}
    _put(result, "manifest_version", _safe_count(manifest.get("manifest_version")))
    _put(result, "analysis_contract", _enum(manifest.get("analysis_contract"), frozenset({"vibedash-evidence-v2"})))
    generated = _timestamp(manifest.get("generated_at"))
    _put(result, "generated_at", generated)

    aggregate: dict[str, Any] = {}
    for key in ("content_sha256", "schema_sha256"):
        _put(aggregate, key, _safe_hash(dataset.get(key)))
    for key in ("source_rows", "analyzed_rows", "column_count"):
        _put(aggregate, key, _safe_count(dataset.get(key)))
    for key in ("truncated", "schema_preview_truncated"):
        _put(aggregate, key, _safe_bool(dataset.get(key)))
    if aggregate:
        result["dataset"] = aggregate

    specification_aggregate: dict[str, Any] = {}
    _put(specification_aggregate, "sha256", _safe_hash(specification.get("sha256")))
    for key in ("metric_count", "chart_count"):
        _put(specification_aggregate, key, _safe_count(specification.get(key)))
    if specification_aggregate:
        result["specification"] = specification_aggregate

    evidence_aggregate: dict[str, Any] = {}
    for key in (
        "insight_count",
        "statistical_test_count",
        "significant_test_count",
        "anomaly_candidate_count",
    ):
        _put(evidence_aggregate, key, _safe_count(evidence.get(key)))

    readiness_aggregate: dict[str, Any] = {}
    _put(readiness_aggregate, "contract", _enum(readiness.get("contract"), _READINESS_CONTRACTS))
    _put(readiness_aggregate, "status", _enum(readiness.get("status"), _READINESS_STATUSES))
    _put(readiness_aggregate, "score", _safe_score(readiness.get("score")))
    for key in ("blocking_issue_count", "warning_count"):
        _put(readiness_aggregate, key, _safe_count(readiness.get(key)))
    if readiness_aggregate:
        evidence_aggregate["readiness"] = readiness_aggregate

    brief_aggregate: dict[str, Any] = {}
    _put(brief_aggregate, "contract", _enum(decision_brief.get("contract"), _DECISION_BRIEF_CONTRACTS))
    _put(brief_aggregate, "status", _enum(decision_brief.get("status"), _DECISION_BRIEF_STATUSES))
    _put(brief_aggregate, "priority_count", _safe_count(decision_brief.get("priority_count")))
    if brief_aggregate:
        evidence_aggregate["decision_brief"] = brief_aggregate
    if evidence_aggregate:
        result["evidence"] = evidence_aggregate
    return result


def _comparison_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": "comparison"}
    _put(result, "manifest_version", _safe_count(manifest.get("manifest_version")))
    _put(result, "analysis_contract", _enum(manifest.get("analysis_contract"), frozenset({"vibedash-period-comparison-v1"})))
    generated = _timestamp(manifest.get("generated_at"))
    _put(result, "generated_at", generated)

    inputs = _mapping(manifest.get("inputs"))
    safe_inputs: dict[str, Any] = {}
    for name in ("baseline", "current"):
        source = _mapping(inputs.get(name))
        item: dict[str, Any] = {}
        for key in ("content_sha256", "schema_sha256"):
            _put(item, key, _safe_hash(source.get(key)))
        for key in ("source_rows", "analyzed_rows", "column_count"):
            _put(item, key, _safe_count(source.get(key)))
        _put(item, "schema_preview_truncated", _safe_bool(source.get("schema_preview_truncated")))
        if item:
            safe_inputs[name] = item
    if safe_inputs:
        result["inputs"] = safe_inputs

    comparison = _mapping(manifest.get("comparison"))
    aggregate: dict[str, Any] = {}
    _put(aggregate, "contract", _enum(comparison.get("contract"), _COMPARISON_CONTRACTS))
    _put(aggregate, "result_sha256", _safe_hash(comparison.get("result_sha256")))
    _put(aggregate, "status", _enum(comparison.get("status"), _COMPARISON_STATUSES))
    counts = _mapping(comparison.get("counts"))
    safe_counts: dict[str, Any] = {}
    for key in (
        "eligible_metrics",
        "tested_metrics",
        "displayed_metrics",
        "significant_metrics",
        "drift_signals",
    ):
        _put(safe_counts, key, _safe_count(counts.get(key)))
    if safe_counts:
        aggregate["counts"] = safe_counts
    schema_changes = _mapping(comparison.get("schema_change_counts"))
    safe_schema_changes: dict[str, Any] = {}
    for key in ("missing", "new", "type"):
        _put(safe_schema_changes, key, _safe_count(schema_changes.get(key)))
    if safe_schema_changes:
        aggregate["schema_change_counts"] = safe_schema_changes
    if aggregate:
        result["comparison"] = aggregate
    return result


def _safe_manifest(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    contract = value.get("analysis_contract")
    if contract == "vibedash-period-comparison-v1" or "inputs" in value or "comparison" in value:
        return _comparison_manifest(value)
    return _normal_manifest(value)


def _safe_account(account: Any) -> dict[str, Any]:
    source = _mapping(account)
    result: dict[str, Any] = {}
    email = _text(source.get("email"), _EMAIL_MAX)
    _put(result, "email", email)
    for key in ("created_at", "last_login_at"):
        _put(result, key, _timestamp(source.get(key)))
    return result


def _safe_job(job: Any) -> dict[str, Any] | None:
    if not isinstance(job, Mapping):
        return None
    result: dict[str, Any] = {}
    _put(result, "id", _safe_id(job.get("id")))
    _put(result, "status", _enum(job.get("status"), _JOB_STATUSES))
    for key in ("created_at", "updated_at", "started_at", "completed_at"):
        _put(result, key, _timestamp(job.get(key)))
    _put(result, "error_code", _enum(job.get("error_code"), _ERROR_CODES))
    _put(result, "analysis_kind", _enum(job.get("analysis_kind"), _ANALYSIS_KINDS))
    # The durable job stores this in payload.  Never copy the payload itself.
    payload = _mapping(job.get("payload"))
    if "analysis_kind" not in result:
        _put(result, "analysis_kind", _enum(payload.get("analysis_kind"), _ANALYSIS_KINDS))
    demo = job.get("demo")
    if demo is None:
        demo = bool(payload.get("demo_dataset")) if "demo_dataset" in payload else None
    _put(result, "demo", _safe_bool(demo))
    session_present = job.get("session_present")
    if session_present is None and "session_id" in job:
        session_present = bool(job.get("session_id"))
    _put(result, "session_present", _safe_bool(session_present))
    manifest = _safe_manifest(job.get("manifest"))
    if manifest:
        result["manifest"] = manifest
    return result


def _safe_evidence(value: Any) -> dict[str, Any] | None:
    source = _mapping(value)
    result: dict[str, Any] = {}
    _put(result, "contract", _enum(source.get("contract"), _DECISION_CASE_CONTRACTS))
    _put(result, "analysis_job_id", _safe_id(source.get("analysis_job_id")))
    _put(result, "analysis_contract", _enum(source.get("analysis_contract"), _ANALYSIS_CONTRACTS))
    _put(result, "dataset_sha256", _safe_hash(source.get("dataset_sha256")))
    priority = _mapping(source.get("priority"))
    safe_priority: dict[str, Any] = {}
    _put(safe_priority, "number", _safe_int(priority.get("number"), minimum=1, maximum=3))
    # Category, title, finding, actions and evidence prose may include source
    # data and are deliberately excluded.  The priority number is the only
    # stable, non-content field needed by the export consumer.
    if safe_priority:
        result["priority"] = safe_priority
    return result or None


def _safe_case(case: Any) -> dict[str, Any] | None:
    if not isinstance(case, Mapping):
        return None
    result: dict[str, Any] = {}
    _put(result, "id", _safe_id(case.get("id")))
    _put(result, "job_id", _safe_id(case.get("job_id")))
    _put(result, "priority", _safe_int(case.get("priority"), minimum=1, maximum=3))
    _put(result, "status", _enum(case.get("status"), _DECISION_STATUSES))
    for key, maximum in (
        ("owner", 120),
        ("decision", 1000),
        ("success_metric", 240),
        ("target_outcome", 500),
        ("actual_outcome", 1200),
        ("review_date", 10),
    ):
        value = (
            _review_date(case.get(key))
            if key == "review_date"
            else _text(case.get(key), maximum, allow_empty=key == "actual_outcome")
        )
        _put(result, key, value)
    for key in ("created_at", "updated_at", "resolved_at"):
        _put(result, key, _timestamp(case.get(key)))
    evidence = _safe_evidence(case.get("evidence_snapshot"))
    if evidence:
        result["evidence"] = evidence
    return result


def build_account_export(
    account: Mapping[str, Any] | None,
    jobs: Sequence[Mapping[str, Any]] | None,
    *,
    jobs_truncated: bool,
    decision_cases: Sequence[Mapping[str, Any]] | None,
    decisions_truncated: bool | None = None,
    decision_cases_truncated: bool | None = None,
    generated_at: Any = None,
) -> dict[str, Any]:
    """Build a bounded account export from caller-supplied records only."""
    if isinstance(jobs, (str, bytes, bytearray)) or (jobs is not None and not isinstance(jobs, Sequence)):
        raise AccountExportError("Invalid account export jobs collection.")
    if isinstance(decision_cases, (str, bytes, bytearray)) or (decision_cases is not None and not isinstance(decision_cases, Sequence)):
        raise AccountExportError("Invalid account export decisions collection.")
    if decisions_truncated is None:
        decisions_truncated = decision_cases_truncated
    if not isinstance(jobs_truncated, bool) or not isinstance(decisions_truncated, bool):
        raise AccountExportError("Invalid account export truncation flag.")
    generated = _generated_timestamp(generated_at) if generated_at is not None else datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    if generated is None:
        raise AccountExportError("Invalid account export timestamp.")

    job_count = len(jobs) if jobs is not None else 0
    case_count = len(decision_cases) if decision_cases is not None else 0
    safe_jobs = [_safe_job(item) for item in (jobs[:MAX_ACCOUNT_EXPORT_JOBS] if jobs is not None else [])]
    safe_cases = [_safe_case(item) for item in (decision_cases[:MAX_ACCOUNT_EXPORT_CASES] if decision_cases is not None else [])]
    payload: dict[str, Any] = {
        "contract": ACCOUNT_EXPORT_CONTRACT,
        "generated_at": generated,
        "account": _safe_account(account),
        "jobs": [item for item in safe_jobs if item is not None],
        "jobs_truncated": jobs_truncated or job_count > MAX_ACCOUNT_EXPORT_JOBS,
        "decision_cases": [item for item in safe_cases if item is not None],
        "decisions_truncated": decisions_truncated or case_count > MAX_ACCOUNT_EXPORT_CASES,
        "pilot_metrics": {"included": False, "reason": "pilot metrics are excluded from account exports"},
    }
    # Validate the contract at construction time too, so callers never receive
    # a payload that the serializer cannot safely encode.
    try:
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AccountExportError("Account export contains unsupported data.") from error
    return payload


def serialize_account_export(
    payload: Mapping[str, Any], *, max_bytes: int = MAX_ACCOUNT_EXPORT_BYTES
) -> bytes:
    """Serialize an account export deterministically and enforce its byte cap."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise AccountExportError("Invalid account export byte limit.")
    if not isinstance(payload, Mapping):
        raise AccountExportError("Invalid account export payload.")
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise AccountExportError("Account export contains unsupported data.") from error
    if len(encoded) > max_bytes:
        raise AccountExportTooLargeError("Account export exceeds the byte limit.")
    return encoded


__all__ = [
    "ACCOUNT_EXPORT_CONTRACT",
    "MAX_ACCOUNT_EXPORT_JOBS",
    "MAX_ACCOUNT_EXPORT_CASES",
    "MAX_ACCOUNT_EXPORT_BYTES",
    "AccountExportError",
    "AccountExportTooLargeError",
    "AccountExportSizeError",
    "build_account_export",
    "serialize_account_export",
]
