"""Durable lifecycle storage and bounded execution for VibeDash analyses."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator


JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_HISTORY_JOBS = 50


class AnalysisJobCapacityError(RuntimeError):
    """The bounded single-instance analysis queue has reached capacity."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _validated_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid analysis job identifier.")
    return job_id


def _validated_scope_id(scope_id: str) -> str:
    if not isinstance(scope_id, str) or not JOB_ID_PATTERN.fullmatch(scope_id):
        raise ValueError("Invalid analysis scope identifier.")
    return scope_id


def _serialized_json(value: Dict[str, Any]) -> str:
    if not isinstance(value, dict):
        raise ValueError("Analysis metadata must be an object.")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


class AnalysisJobStore:
    """Persist analysis jobs and enforce atomic state transitions."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(
        self,
        scope_id: str,
        payload: Dict[str, Any],
        *,
        max_active_per_scope: int = 2,
        max_active_total: int = 25,
    ) -> Dict[str, Any]:
        normalized_scope_id = _validated_scope_id(scope_id)
        if not isinstance(max_active_per_scope, int) or max_active_per_scope < 1:
            raise ValueError("max_active_per_scope must be a positive integer.")
        if not isinstance(max_active_total, int) or max_active_total < 1:
            raise ValueError("max_active_total must be a positive integer.")

        payload_json = _serialized_json(payload)
        job_id = uuid.uuid4().hex
        created_at = _utc_now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active_for_scope = connection.execute(
                """
                SELECT COUNT(*) FROM analysis_jobs
                WHERE scope_id = ? AND status IN ('queued', 'running')
                """,
                (normalized_scope_id,),
            ).fetchone()[0]
            active_total = connection.execute(
                """
                SELECT COUNT(*) FROM analysis_jobs
                WHERE status IN ('queued', 'running')
                """
            ).fetchone()[0]
            if (
                active_for_scope >= max_active_per_scope
                or active_total >= max_active_total
            ):
                raise AnalysisJobCapacityError(
                    "The analysis queue is currently at capacity."
                )
            connection.execute(
                """
                INSERT INTO analysis_jobs (
                    id, scope_id, status, payload_json, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    normalized_scope_id,
                    payload_json,
                    created_at,
                    created_at,
                ),
            )
        return self.get(job_id, normalized_scope_id)

    def get(self, job_id: str, scope_id: str | None = None) -> Dict[str, Any] | None:
        normalized_id = _validated_job_id(job_id)
        parameters: tuple[Any, ...]
        scope_filter = ""
        if scope_id is None:
            parameters = (normalized_id,)
        else:
            normalized_scope_id = _validated_scope_id(scope_id)
            scope_filter = " AND scope_id = ?"
            parameters = (normalized_id, normalized_scope_id)

        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT id, scope_id, status, payload_json, manifest_json,
                       session_id, error_code, created_at, updated_at,
                       started_at, completed_at
                FROM analysis_jobs
                WHERE id = ?{scope_filter}
                """,
                parameters,
            ).fetchone()
        return self._row_to_job(row)

    def list_for_scope(
        self,
        scope_id: str,
        *,
        limit: int = 20,
    ) -> list[Dict[str, Any]]:
        """Return recent jobs for one signed browser scope."""
        normalized_scope_id = _validated_scope_id(scope_id)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer.")
        if not 1 <= limit <= MAX_HISTORY_JOBS:
            raise ValueError(
                f"limit must be between 1 and {MAX_HISTORY_JOBS}."
            )
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, scope_id, status, payload_json, manifest_json,
                       session_id, error_code, created_at, updated_at,
                       started_at, completed_at
                FROM analysis_jobs
                WHERE scope_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (normalized_scope_id, limit),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def claim(self, job_id: str) -> Dict[str, Any] | None:
        """Atomically move one queued job to running."""
        normalized_id = _validated_job_id(job_id)
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'running', started_at = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (timestamp, timestamp, normalized_id),
            )
            if cursor.rowcount != 1:
                return None
        return self.get(normalized_id)

    def complete(
        self,
        job_id: str,
        session_id: str,
        manifest: Dict[str, Any] | None = None,
    ) -> bool:
        normalized_id = _validated_job_id(job_id)
        try:
            normalized_session_id = str(uuid.UUID(session_id))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Invalid dashboard session identifier.") from error
        manifest_json = _serialized_json(manifest or {})
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_jobs SET status = 'completed', session_id = ?,
                    manifest_json = ?, error_code = NULL, completed_at = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    normalized_session_id,
                    manifest_json,
                    timestamp,
                    timestamp,
                    normalized_id,
                ),
            )
        return cursor.rowcount == 1

    def fail(self, job_id: str, error_code: str = "analysis_failed") -> bool:
        normalized_id = _validated_job_id(job_id)
        if not isinstance(error_code, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{0,63}", error_code
        ):
            raise ValueError("Invalid analysis error code.")
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'failed', error_code = ?, completed_at = ?,
                    updated_at = ?
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (error_code, timestamp, timestamp, normalized_id),
            )
        return cursor.rowcount == 1

    def fail_stale_running(self, timeout_seconds: int) -> int:
        if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer.")
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
        ).isoformat(timespec="milliseconds")
        timestamp = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'failed', error_code = 'worker_interrupted',
                    completed_at = ?, updated_at = ?
                WHERE status = 'running' AND started_at < ?
                """,
                (timestamp, timestamp, cutoff),
            )
        return cursor.rowcount

    def purge_terminal(self, retention_hours: int) -> int:
        if not isinstance(retention_hours, int) or retention_hours < 1:
            raise ValueError("retention_hours must be a positive integer.")
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        ).isoformat(timespec="milliseconds")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM analysis_jobs
                WHERE status IN ('completed', 'failed') AND updated_at < ?
                """,
                (cutoff,),
            )
        return cursor.rowcount

    @staticmethod
    def _row_to_job(row: sqlite3.Row | None) -> Dict[str, Any] | None:
        if row is None:
            return None
        job = dict(row)
        job["payload"] = json.loads(job.pop("payload_json"))
        manifest_json = job.pop("manifest_json", None)
        job["manifest"] = json.loads(manifest_json) if manifest_json else None
        return job

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    scope_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed')
                    ),
                    payload_json TEXT NOT NULL,
                    manifest_json TEXT,
                    session_id TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_analysis_jobs_scope_time
                ON analysis_jobs(scope_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status_time
                ON analysis_jobs(status, created_at ASC);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(analysis_jobs)"
                ).fetchall()
            }
            if "manifest_json" not in columns:
                connection.execute(
                    "ALTER TABLE analysis_jobs ADD COLUMN manifest_json TEXT"
                )


class AnalysisJobDispatcher:
    """Run a bounded number of analysis jobs outside request threads."""

    def __init__(self, max_workers: int = 1) -> None:
        if not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer.")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="analysis-job",
        )
        self._scheduled: set[str] = set()
        self._lock = threading.Lock()

    def submit(
        self,
        app,
        job_id: str,
        processor: Callable[[Dict[str, Any]], str | Dict[str, Any]],
    ) -> bool:
        normalized_id = _validated_job_id(job_id)
        with self._lock:
            if normalized_id in self._scheduled:
                return False
            self._scheduled.add(normalized_id)
        self._executor.submit(
            self._run,
            app,
            normalized_id,
            processor,
        )
        return True

    def _run(self, app, job_id: str, processor) -> None:
        try:
            with app.app_context():
                store = AnalysisJobStore(app.config["VIBEDASH_JOB_STORE_PATH"])
                job = store.claim(job_id)
                if job is None:
                    return
                app.logger.info(
                    "VibeDash analysis job started",
                    extra={"event": "vibedash_job_started"},
                )
                try:
                    result = processor(job)
                    if isinstance(result, str):
                        session_id = result
                        manifest = None
                    elif isinstance(result, dict):
                        session_id = result.get("session_id")
                        manifest = result.get("manifest")
                    else:
                        raise RuntimeError("Analysis processor returned no result.")
                    if not store.complete(job_id, session_id, manifest):
                        raise RuntimeError("Analysis job completion transition failed.")
                except Exception:
                    store.fail(job_id)
                    app.logger.exception(
                        "VibeDash analysis job failed",
                        extra={"event": "vibedash_job_failed"},
                    )
                else:
                    app.logger.info(
                        "VibeDash analysis job completed",
                        extra={"event": "vibedash_job_completed"},
                    )
        finally:
            with self._lock:
                self._scheduled.discard(job_id)
