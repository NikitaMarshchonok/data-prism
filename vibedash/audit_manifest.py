"""Build bounded, privacy-conscious audit metadata for VibeDash analyses."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MANIFEST_VERSION = 2
ANALYSIS_CONTRACT = "vibedash-evidence-v2"
MAX_SCHEMA_PREVIEW_COLUMNS = 50

COMPARISON_MANIFEST_VERSION = 1
COMPARISON_ANALYSIS_CONTRACT = "vibedash-period-comparison-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    source_path = Path(path)
    digest = hashlib.sha256()
    with source_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_audit_manifest(
    *,
    dataset_path: str | Path,
    dataframe,
    source_row_count: int,
    truncated: bool,
    source_name: str,
    prompt: str,
    demo_dataset: str,
    run_id: str | None,
    viz_spec: Mapping[str, Any],
    dashboard_data: Mapping[str, Any],
    engine_version: str,
    retention_hours: int,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Return reproducibility metadata without including source row values."""
    if source_row_count < len(dataframe):
        raise ValueError("source_row_count cannot be smaller than analyzed rows.")

    generated = generated_at or datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    generated = generated.astimezone(timezone.utc)

    schema = [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(dataframe.columns, dataframe.dtypes)
    ]
    schema_preview = schema[:MAX_SCHEMA_PREVIEW_COLUMNS]
    statistical_validation = dashboard_data.get("statistical_validation") or {}
    pattern_analysis = dashboard_data.get("pattern_analysis") or {}
    anomaly_detection = pattern_analysis.get("anomaly_detection") or {}
    readiness = dashboard_data.get("readiness") or {}
    decision_brief = dashboard_data.get("decision_brief") or {}

    normalized_spec = dict(viz_spec)
    return {
        "manifest_version": MANIFEST_VERSION,
        "analysis_contract": ANALYSIS_CONTRACT,
        "engine_version": str(engine_version)[:128],
        "generated_at": generated.isoformat(timespec="milliseconds"),
        "run": {"id": run_id},
        "dataset": {
            "source_name": str(source_name)[:255],
            "content_sha256": _file_sha256(dataset_path),
            "schema_sha256": _canonical_sha256(schema),
            "source_rows": int(source_row_count),
            "analyzed_rows": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "truncated": bool(truncated),
            "schema_preview": schema_preview,
            "schema_preview_truncated": len(schema) > len(schema_preview),
        },
        "request": {
            "prompt": prompt,
            "sha256": _canonical_sha256({"prompt": prompt}),
            "demo_dataset": demo_dataset or None,
        },
        "specification": {
            "sha256": _canonical_sha256(normalized_spec),
            "title": str(normalized_spec.get("title", "Analysis"))[:300],
            "metric_count": len(normalized_spec.get("metrics") or []),
            "chart_count": len(normalized_spec.get("charts") or []),
        },
        "evidence": {
            "insight_count": len(dashboard_data.get("insights") or []),
            "statistical_test_count": len(
                statistical_validation.get("tests") or []
            ),
            "significant_test_count": sum(
                bool(test.get("significant"))
                for test in statistical_validation.get("tests") or []
                if isinstance(test, Mapping)
            ),
            "anomaly_candidate_count": len(
                anomaly_detection.get("top_anomalies") or []
            ),
            "readiness": {
                "contract": readiness.get("contract"),
                "status": readiness.get("status"),
                "score": readiness.get("score"),
                "blocking_issue_count": (readiness.get("counts") or {}).get(
                    "blocked", 0
                ),
                "warning_count": (readiness.get("counts") or {}).get(
                    "warnings", 0
                ),
            },
            "decision_brief": {
                "contract": decision_brief.get("contract"),
                "status": decision_brief.get("status"),
                "priority_count": decision_brief.get("priority_count", 0),
            },
        },
        "retention": {
            "temporary": True,
            "hours": int(retention_hours),
        },
    }


def build_period_comparison_manifest(
    *,
    baseline_path: str | Path,
    current_path: str | Path,
    baseline_dataframe,
    current_dataframe,
    baseline_filename: str,
    current_filename: str,
    baseline_label: str,
    current_label: str,
    comparison_result: Mapping[str, Any],
    run_id: str | None,
    engine_version: str,
    retention_hours: int,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build aggregate-only reproducibility metadata for a period comparison.

    This deliberately accepts already-loaded frames only to derive schema and
    counts.  No source rows, categorical values, or full result payload are
    copied into the manifest.
    """
    paths = (Path(baseline_path), Path(current_path))
    for source_path in paths:
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("Comparison input must be a regular file.")

    def _schema(frame):
        schema = [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in zip(frame.columns, frame.dtypes)
        ]
        return {
            "sha256": _canonical_sha256(schema),
            "column_count": int(len(schema)),
            "schema_preview": schema[:MAX_SCHEMA_PREVIEW_COLUMNS],
            "schema_preview_truncated": len(schema) > MAX_SCHEMA_PREVIEW_COLUMNS,
        }

    def _input(path, frame, filename, label):
        if not isinstance(filename, str) or not filename:
            raise ValueError("Comparison input filename is invalid.")
        if not isinstance(label, str) or not label:
            raise ValueError("Comparison input label is invalid.")
        return {
            "filename": filename[:255],
            "label": label[:80],
            "content_sha256": _file_sha256(path),
            "source_rows": int(len(frame)),
            "analyzed_rows": int(len(frame)),
            **_schema(frame),
        }

    generated = generated_at or datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    generated = generated.astimezone(timezone.utc)
    result = comparison_result if isinstance(comparison_result, Mapping) else {}
    numeric = result.get("numeric_metrics") or {}
    drift = result.get("distribution_drift") or {}
    schema_changes = result.get("schema_changes") or {}
    # Hash the complete aggregate result for reproducibility; it contains no
    # rows and the engine contract excludes category values.
    result_hash = _canonical_sha256(result)
    manifest = {
        "manifest_version": COMPARISON_MANIFEST_VERSION,
        "analysis_contract": COMPARISON_ANALYSIS_CONTRACT,
        "engine_version": str(engine_version)[:128],
        "generated_at": generated.isoformat(timespec="milliseconds"),
        "run": {"id": run_id},
        "inputs": {
            "baseline": _input(paths[0], baseline_dataframe, baseline_filename, baseline_label),
            "current": _input(paths[1], current_dataframe, current_filename, current_label),
        },
        "comparison": {
            "contract": str(result.get("contract", ""))[:128],
            "result_sha256": result_hash,
            "status": str(result.get("status", ""))[:64],
            "baseline_label": baseline_label[:80],
            "current_label": current_label[:80],
            "schema_change_counts": {
                "missing": len(schema_changes.get("missing_columns") or []),
                "new": len(schema_changes.get("new_columns") or []),
                "type": len(schema_changes.get("type_changes") or []),
            },
            "counts": {
                "eligible_metrics": int(numeric.get("eligible_count", 0)),
                "tested_metrics": int(numeric.get("tested_count", 0)),
                "displayed_metrics": int(numeric.get("display_count", 0)),
                "significant_metrics": int(numeric.get("significant_count", 0)),
                "drift_signals": len(drift.get("signals") or []),
            },
        },
        "retention": {"temporary": True, "hours": int(retention_hours)},
    }
    # Keep this boundary strict so a future engine change cannot persist NaN,
    # infinity, or an unserializable object in the audit record.
    json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return manifest
