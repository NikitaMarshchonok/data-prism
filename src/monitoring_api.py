"""Authenticated machine-readable endpoints for data-drift monitoring."""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import uuid
from pathlib import Path
from typing import Tuple

import pandas as pd
from flask import Blueprint, current_app, g, jsonify, request
from werkzeug.utils import secure_filename

from src.data_drift import (
    compare_to_baseline,
    create_baseline_profile,
    load_baseline_profile,
    save_baseline_profile,
)
from src.data_loader import is_supported_data_file, load_data
from src.drift_store import DriftStore


monitoring_api_bp = Blueprint("monitoring_api", __name__, url_prefix="/api/v1/drift")
IDENTIFIER_PATTERN = re.compile(r"[0-9a-f]{32}")


@monitoring_api_bp.before_request
def authenticate_monitoring_api():
    """Require an API key and derive a non-secret isolated storage scope."""
    expected_key = current_app.config.get("DATA_PRISM_API_KEY")
    if not isinstance(expected_key, str) or not expected_key:
        return _error_response(
            "api_not_configured",
            "Monitoring API is disabled until DATA_PRISM_API_KEY is configured.",
            503,
        )
    if len(expected_key) < 32:
        return _error_response(
            "api_key_too_short",
            "DATA_PRISM_API_KEY must contain at least 32 characters.",
            503,
        )
    provided_key = request.headers.get("X-API-Key") or _bearer_token()
    if not isinstance(provided_key, str) or not secrets.compare_digest(
        provided_key,
        expected_key,
    ):
        return _error_response("unauthorized", "A valid API key is required.", 401)
    key_hash = hashlib.sha256(expected_key.encode("utf-8")).hexdigest()
    g.monitoring_scope_id = f"api-{key_hash[:32]}"
    return None


@monitoring_api_bp.post("/baselines")
def create_api_baseline():
    try:
        data, truncated, dataset_name, _content_hash = _load_transient_dataset()
        profile = create_baseline_profile(data, dataset_name=dataset_name)
        baseline_id = uuid.uuid4().hex
        save_baseline_profile(profile, _api_baseline_path(baseline_id))
    except ValueError as error:
        return _error_response("invalid_dataset", str(error), 400)
    except OSError:
        current_app.logger.exception("Failed to persist API baseline")
        return _error_response("storage_error", "Could not persist baseline profile.", 500)

    return (
        jsonify(
            {
                "baseline_id": baseline_id,
                "created_at": profile["created_at"],
                "dataset_name": dataset_name,
                "row_count": profile["row_count"],
                "column_count": profile["column_count"],
                "columns_truncated": profile["columns_truncated"],
                "rows_truncated": truncated,
                "stores_raw_rows": False,
            }
        ),
        201,
    )


@monitoring_api_bp.post("/checks")
def create_api_check():
    baseline_id = request.form.get("baseline_id", "")
    if not IDENTIFIER_PATTERN.fullmatch(baseline_id):
        return _error_response("invalid_baseline_id", "baseline_id is invalid.", 400)
    baseline_path = _api_baseline_path(baseline_id)
    if not baseline_path.exists():
        return _error_response("baseline_not_found", "Baseline was not found.", 404)

    try:
        baseline_profile = load_baseline_profile(baseline_path)
        data, truncated, dataset_name, content_hash = _load_transient_dataset()
        report = compare_to_baseline(data, baseline_profile)
        idempotency_key = request.headers.get("Idempotency-Key")
        batch_id = _batch_id(baseline_id, content_hash, idempotency_key)
        recorded = _store().record_run(
            report,
            batch_id=batch_id,
            dataset_name=dataset_name,
        )
    except ValueError as error:
        return _error_response("invalid_request", str(error), 400)
    except (OSError, sqlite3.Error):
        current_app.logger.exception("Failed to run API drift check")
        return _error_response("monitoring_error", "Drift check could not be completed.", 500)

    return (
        jsonify(
            {
                "run": recorded,
                "report": report,
                "rows_truncated": truncated,
                "deduplication": (
                    "Idempotency-Key" if idempotency_key else "uploaded content SHA-256"
                ),
            }
        ),
        201 if recorded["created"] else 200,
    )


@monitoring_api_bp.get("/runs")
def list_api_runs():
    try:
        runs = _store().list_runs(limit=_request_limit())
    except ValueError as error:
        return _error_response("invalid_limit", str(error), 400)
    except (OSError, sqlite3.Error):
        current_app.logger.exception("Failed to list API drift runs")
        return _error_response("storage_error", "Could not read drift history.", 500)
    return jsonify({"runs": runs})


@monitoring_api_bp.get("/runs/<run_id>")
def get_api_run(run_id):
    try:
        run = _store().get_run(run_id)
    except ValueError as error:
        return _error_response("invalid_run_id", str(error), 400)
    except (OSError, sqlite3.Error):
        current_app.logger.exception("Failed to read API drift run")
        return _error_response("storage_error", "Could not read drift run.", 500)
    if run is None:
        return _error_response("run_not_found", "Drift run was not found.", 404)
    return jsonify({"run": run})


@monitoring_api_bp.get("/alerts")
def list_api_alerts():
    include_acknowledged = request.args.get("include_acknowledged", "false").lower() == "true"
    try:
        alerts = _store().list_alerts(
            unacknowledged_only=not include_acknowledged,
            limit=_request_limit(),
        )
    except ValueError as error:
        return _error_response("invalid_limit", str(error), 400)
    except (OSError, sqlite3.Error):
        current_app.logger.exception("Failed to list API drift alerts")
        return _error_response("storage_error", "Could not read drift alerts.", 500)
    return jsonify({"alerts": alerts})


@monitoring_api_bp.post("/alerts/<int:alert_id>/acknowledge")
def acknowledge_api_alert(alert_id):
    try:
        acknowledged = _store().acknowledge_alert(alert_id)
    except ValueError as error:
        return _error_response("invalid_alert_id", str(error), 400)
    except (OSError, sqlite3.Error):
        current_app.logger.exception("Failed to acknowledge API drift alert")
        return _error_response("storage_error", "Could not update drift alert.", 500)
    if not acknowledged:
        return _error_response(
            "alert_not_found",
            "Alert was not found or was already acknowledged.",
            404,
        )
    return jsonify({"alert_id": alert_id, "acknowledged": True})


def _load_transient_dataset() -> Tuple[pd.DataFrame, bool, str, str]:
    uploaded_file = request.files.get("datafile")
    if uploaded_file is None or not uploaded_file.filename:
        raise ValueError("A datafile upload is required.")
    dataset_name = secure_filename(uploaded_file.filename)
    if not dataset_name or not is_supported_data_file(dataset_name):
        raise ValueError("The uploaded data format is not supported.")

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(dataset_name).suffix.lower()
    temporary_path = upload_folder / f"api-{uuid.uuid4().hex}{suffix}"
    try:
        uploaded_file.save(temporary_path)
        content_hash = _file_sha256(temporary_path)
        data, truncated = load_data(temporary_path)
        if data is None:
            raise ValueError("The uploaded dataset could not be read.")
        return data, truncated, dataset_name, content_hash
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _api_baseline_path(baseline_id: str) -> Path:
    if not IDENTIFIER_PATTERN.fullmatch(baseline_id):
        raise ValueError("baseline_id is invalid.")
    return (
        Path(current_app.config["BASELINE_FOLDER"])
        / "api"
        / g.monitoring_scope_id
        / f"{baseline_id}.json"
    )


def _store() -> DriftStore:
    return DriftStore(
        current_app.config["DRIFT_STORE_PATH"],
        g.monitoring_scope_id,
        retention=current_app.config["DRIFT_HISTORY_RETENTION"],
    )


def _batch_id(baseline_id: str, content_hash: str, idempotency_key: str | None) -> str:
    if idempotency_key is not None:
        if not 1 <= len(idempotency_key) <= 128:
            raise ValueError("Idempotency-Key must contain between 1 and 128 characters.")
        source = idempotency_key
    else:
        source = content_hash
    digest = hashlib.sha256(f"{baseline_id}:{source}".encode("utf-8")).hexdigest()
    return f"api-{digest}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_limit() -> int:
    raw_limit = request.args.get("limit", "20")
    try:
        return int(raw_limit)
    except ValueError as error:
        raise ValueError("limit must be an integer.") from error


def _bearer_token() -> str | None:
    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer":
        return token
    return None


def _error_response(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status
