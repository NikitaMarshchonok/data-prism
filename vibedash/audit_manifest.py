"""Build bounded, privacy-conscious audit metadata for VibeDash analyses."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MANIFEST_VERSION = 1
ANALYSIS_CONTRACT = "vibedash-evidence-v1"
MAX_SCHEMA_PREVIEW_COLUMNS = 50


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
        },
        "retention": {
            "temporary": True,
            "hours": int(retention_hours),
        },
    }
