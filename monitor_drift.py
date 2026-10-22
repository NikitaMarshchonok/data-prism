#!/usr/bin/env python3
"""Cron- and CI-ready command line runner for persistent data-drift checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd

from src.data_drift import (
    MAX_PROFILE_FEATURES,
    compare_to_baseline,
    create_baseline_profile,
    load_baseline_profile,
    save_baseline_profile,
)
from src.data_loader import is_supported_data_file, load_data
from src.drift_store import DriftStore
from src.semantic_contract import (
    MAX_CONTRACT_JSON_BYTES,
    infer_semantic_contract,
    parse_semantic_contract_json,
)


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_THRESHOLD_REACHED = 2
FAIL_ON_VALUES = {"never", "warning", "critical"}
REQUIRED_JOB_FIELDS = {
    "job_id",
    "job_name",
    "baseline_profile",
    "current_data",
    "history_database",
}
OPTIONAL_JOB_FIELDS = {"fail_on", "retention", "current_semantic_contract"}


def propose_semantics(data_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    """Write an unconfirmed proposal for human review, without overwriting files."""
    source = Path(data_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output or output.exists():
        raise ValueError("Semantic proposal output must be a new file, not the source dataset.")
    data, truncated, _hash = _load_stable_dataset(source)
    if len(data.columns) > MAX_PROFILE_FEATURES:
        raise ValueError(f"Confirmed baselines support at most {MAX_PROFILE_FEATURES} columns.")
    proposal = infer_semantic_contract(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(proposal, target, ensure_ascii=False, indent=2, allow_nan=False)
    return {
        "command": "propose-semantics", "semantic_contract": str(output),
        "confirmed": False, "rows_truncated": truncated,
    }


def create_baseline(
    data_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str | None = None,
    semantic_contract_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Create an aggregate baseline profile from a local dataset."""
    source_path = Path(data_path).expanduser().resolve()
    profile_path = Path(output_path).expanduser().resolve()
    if source_path == profile_path:
        raise ValueError("Baseline output must not overwrite the source dataset.")
    if semantic_contract_path is None:
        raise ValueError("A reviewed semantic_contract_path is required to create a baseline.")
    contract_path = Path(semantic_contract_path).expanduser().resolve()
    if contract_path == profile_path:
        raise ValueError("Baseline output must not overwrite the semantic contract.")
    contract = _read_semantic_contract(contract_path)
    data, truncated, _content_hash = _load_stable_dataset(source_path)
    profile = create_baseline_profile(
        data,
        dataset_name=dataset_name or source_path.name,
        semantic_contract=contract,
    )
    save_baseline_profile(profile, profile_path)
    return {
        "command": "create-baseline",
        "baseline_profile": str(profile_path),
        "created_at": profile["created_at"],
        "dataset_name": profile["dataset_name"],
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "profile_version": profile["profile_version"],
        "semantic_sha256": profile["semantic_contract"]["sha256"],
        "rows_truncated": truncated,
        "stores_raw_rows": False,
    }


def run_monitoring_job(
    config_path: str | Path,
    *,
    batch_id: str | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Execute one configured drift check and return its JSON payload and exit code."""
    job = load_job_config(config_path)
    baseline_profile = load_baseline_profile(job["baseline_profile"])
    current_data, truncated, content_hash = _load_stable_dataset(job["current_data"])
    contract = (
        _read_semantic_contract(job["current_semantic_contract"])
        if job.get("current_semantic_contract") is not None else None
    )
    report = compare_to_baseline(current_data, baseline_profile, semantic_contract=contract)
    revision = hashlib.sha256(json.dumps(
        {"baseline": baseline_profile, "semantics": report["semantics"]},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    effective_batch_id = _effective_batch_id(f"{content_hash}:{revision}", batch_id)
    scope_id = _job_scope_id(job["job_id"])
    store = DriftStore(
        job["history_database"],
        scope_id,
        retention=job["retention"],
    )
    recorded = store.record_run(
        report,
        batch_id=effective_batch_id,
        dataset_name=job["current_data"].name,
    )
    threshold_reached = _threshold_reached(report["status"], job["fail_on"])
    payload = {
        "command": "run",
        "job_id": job["job_id"],
        "job_name": job["job_name"],
        "status": report["status"],
        "threshold_reached": threshold_reached,
        "fail_on": job["fail_on"],
        "rows_truncated": truncated,
        "run": recorded,
        "report": report,
    }
    return payload, EXIT_THRESHOLD_REACHED if threshold_reached else EXIT_OK


def load_job_config(config_path: str | Path) -> Dict[str, Any]:
    """Load, validate, and resolve a monitoring job configuration."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Monitoring config does not exist: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Monitoring config is not valid JSON: {error.msg}") from error
    if not isinstance(config, dict):
        raise ValueError("Monitoring config must be a JSON object.")

    missing = REQUIRED_JOB_FIELDS - set(config)
    unknown = set(config) - REQUIRED_JOB_FIELDS - OPTIONAL_JOB_FIELDS
    if missing:
        raise ValueError(f"Monitoring config is missing: {', '.join(sorted(missing))}.")
    if unknown:
        raise ValueError(f"Monitoring config has unknown fields: {', '.join(sorted(unknown))}.")

    job_id = config["job_id"]
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", job_id):
        raise ValueError(
            "job_id must use 1-128 letters, numbers, dots, underscores, or hyphens."
        )
    job_name = config["job_name"]
    if not isinstance(job_name, str) or not 1 <= len(job_name.strip()) <= 128:
        raise ValueError("job_name must contain between 1 and 128 characters.")
    fail_on = config.get("fail_on", "critical")
    if fail_on not in FAIL_ON_VALUES:
        raise ValueError("fail_on must be one of: never, warning, critical.")
    retention = config.get("retention", 100)
    if not isinstance(retention, int) or not 1 <= retention <= 10000:
        raise ValueError("retention must be an integer between 1 and 10000.")

    base_directory = path.parent
    resolved = {
        "job_id": job_id,
        "job_name": job_name.strip(),
        "baseline_profile": _resolve_config_path(base_directory, config["baseline_profile"]),
        "current_data": _resolve_config_path(base_directory, config["current_data"]),
        "history_database": _resolve_config_path(base_directory, config["history_database"]),
        "fail_on": fail_on,
        "retention": retention,
    }
    if not resolved["baseline_profile"].is_file():
        raise ValueError("baseline_profile does not exist.")
    if not resolved["current_data"].is_file():
        raise ValueError("current_data does not exist.")
    protected_inputs = {
        path,
        resolved["baseline_profile"],
        resolved["current_data"],
    }
    if "current_semantic_contract" in config:
        resolved["current_semantic_contract"] = _resolve_config_path(
            base_directory, config["current_semantic_contract"],
        )
        if not resolved["current_semantic_contract"].is_file():
            raise ValueError("current_semantic_contract does not exist.")
        protected_inputs.add(resolved["current_semantic_contract"])
    if resolved["history_database"] in protected_inputs:
        raise ValueError("history_database must not overwrite a config or input file.")
    if resolved["baseline_profile"] == resolved["current_data"]:
        raise ValueError("baseline_profile and current_data must be different files.")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create baselines and run persistent data-drift checks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    proposal_parser = subparsers.add_parser(
        "propose-semantics", help="Write unconfirmed schema metadata for review.",
    )
    proposal_parser.add_argument("--data", required=True)
    proposal_parser.add_argument("--output", required=True)

    baseline_parser = subparsers.add_parser(
        "create-baseline",
        help="Create an aggregate baseline profile from a dataset.",
    )
    baseline_parser.add_argument("--data", required=True, help="Reference dataset path.")
    baseline_parser.add_argument("--output", required=True, help="Baseline JSON output path.")
    baseline_parser.add_argument("--dataset-name", help="Optional display name.")
    baseline_parser.add_argument(
        "--semantic-contract", required=True, help="Reviewed JSON with confirmed=true.",
    )

    run_parser = subparsers.add_parser("run", help="Execute one monitoring job.")
    run_parser.add_argument("--config", required=True, help="Monitoring job JSON path.")
    run_parser.add_argument(
        "--batch-id",
        help="Optional idempotency key; defaults to current content, baseline revision and semantic context.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "propose-semantics":
            payload = propose_semantics(args.data, args.output)
            exit_code = EXIT_OK
        elif args.command == "create-baseline":
            payload = create_baseline(
                args.data,
                args.output,
                dataset_name=args.dataset_name,
                semantic_contract_path=args.semantic_contract,
            )
            exit_code = EXIT_OK
        else:
            payload, exit_code = run_monitoring_job(
                args.config,
                batch_id=args.batch_id,
            )
    except Exception as error:
        payload = {
            "command": args.command,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        exit_code = EXIT_ERROR
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return exit_code


def _read_semantic_contract(path: Path):
    with path.open("r", encoding="utf-8") as source:
        return parse_semantic_contract_json(source.read(MAX_CONTRACT_JSON_BYTES + 1))


def _load_stable_dataset(path: Path) -> Tuple[pd.DataFrame, bool, str]:
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    if not is_supported_data_file(path.name):
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    checksum_before = _file_sha256(path)
    with redirect_stdout(io.StringIO()):
        data, truncated = load_data(str(path))
    checksum_after = _file_sha256(path)
    if checksum_before != checksum_after:
        raise RuntimeError("Dataset changed while it was being read; retry the monitoring run.")
    if data is None:
        raise ValueError(f"Dataset could not be read: {path}")
    return data, truncated, checksum_after


def _resolve_config_path(base_directory: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Monitoring paths must be non-empty strings.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_directory / path).resolve()


def _effective_batch_id(content_hash: str, batch_id: str | None) -> str:
    if batch_id is not None:
        if not 1 <= len(batch_id) <= 128:
            raise ValueError("batch_id must contain between 1 and 128 characters.")
        source = batch_id
    else:
        source = content_hash
    return f"job-{hashlib.sha256(source.encode('utf-8')).hexdigest()}"


def _job_scope_id(job_id: str) -> str:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    return f"job-{digest[:32]}"


def _threshold_reached(status: str, fail_on: str) -> bool:
    if fail_on == "never":
        return False
    if fail_on == "warning":
        return status in {"warning", "critical"}
    return status == "critical"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
