"""Durable lifecycle storage and bounded execution for VibeDash analyses."""

from __future__ import annotations

import json
import inspect
import re
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

from .pilot_metrics import initialize_metrics, mark_stage, purge_metrics, register_analysis


JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
UPLOAD_FILENAME_PATTERN = re.compile(r"^vibedash-[0-9a-f]{32}\.csv$")
PILOT_SCOPE_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_HISTORY_JOBS = 50


class AnalysisJobCapacityError(RuntimeError):
    """The bounded single-instance analysis queue has reached capacity."""


class ScopeHasActiveJobsError(RuntimeError):
    """A scope cannot be erased while queued or running jobs remain."""


class ScopeClosedError(RuntimeError):
    """The scope has been durably fenced against new persisted work."""


class ScopeDeletionJobs(list):
    """Owned jobs supplied to a scope-artifact cleanup callback.

    ``foreign_scope_upload_filenames`` contains only validated upload names
    still referenced by jobs belonging to another scope.  It is attached to
    the list rather than passed as a second positional argument so existing
    one-argument cleanup callbacks remain source-compatible.  If
    ``foreign_scope_payloads_unknown`` is true, at least one foreign payload
    was unreadable and a file cleanup helper should fail closed for uploads.
    ``foreign_scope_session_ids`` contains validated export session references
    held by other scopes so cleanup can protect those exports too.
    """

    def __init__(
        self,
        jobs: list[Dict[str, Any]],
        foreign_scope_upload_filenames: set[str],
        *,
        foreign_scope_payloads_unknown: bool = False,
        foreign_scope_session_ids: set[str] | None = None,
    ) -> None:
        super().__init__(jobs)
        self.foreign_scope_upload_filenames = frozenset(
            foreign_scope_upload_filenames
        )
        # Short aliases make the safety contract discoverable to callers.
        self.protected_upload_filenames = self.foreign_scope_upload_filenames
        self.protected_filenames = self.foreign_scope_upload_filenames
        self.foreign_scope_payloads_unknown = foreign_scope_payloads_unknown
        self.foreign_scope_session_ids = frozenset(
            foreign_scope_session_ids or ()
        )


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


def _upload_filenames(value: Any) -> set[str]:
    """Return safe, basename-only upload references from persisted JSON."""
    names: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "stored_filename" and isinstance(item, str):
                if UPLOAD_FILENAME_PATTERN.fullmatch(item):
                    names.add(item)
            else:
                names.update(_upload_filenames(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_upload_filenames(item))
    return names


def _validated_pilot_scope_token(token: str | None) -> str | None:
    if token is None:
        return None
    if not isinstance(token, str) or not PILOT_SCOPE_TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Invalid pilot measurement token.")
    return token


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
        pilot_scope_token: str | None = None,
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
            if connection.execute(
                "SELECT 1 FROM deleted_scopes WHERE scope_id = ?",
                (normalized_scope_id,),
            ).fetchone() is not None:
                raise ScopeClosedError("This scope has been closed.")
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
            register_analysis(
                connection, job_id, pilot_scope_token,
                'demo' if payload.get('demo_dataset') else 'upload', created_at,
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
        """Return recent jobs for one signed VibeDash scope."""
        items, _has_more = self.list_for_scope_page(scope_id, limit=limit)
        return items

    def list_for_scope_page(
        self,
        scope_id: str,
        *,
        limit: int,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[list[Dict[str, Any]], bool]:
        """Return one recent-job page and whether another row exists."""
        normalized_scope_id = _validated_scope_id(scope_id)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer.")
        if not 1 <= limit <= MAX_HISTORY_JOBS:
            raise ValueError(
                f"limit must be between 1 and {MAX_HISTORY_JOBS}."
            )
        if connection is not None:
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
                (normalized_scope_id, limit + 1),
            ).fetchall()
        else:
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
                    (normalized_scope_id, limit + 1),
                ).fetchall()
        has_more = len(rows) > limit
        items = rows[:limit]
        return [self._row_to_job(row) for row in items], has_more

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
            mark_stage(connection, normalized_id, 'started_at', timestamp)
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
            if cursor.rowcount == 1:
                mark_stage(connection, normalized_id, 'completed_at', timestamp)
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
            if cursor.rowcount == 1:
                mark_stage(connection, normalized_id, 'failed_at', timestamp)
        return cursor.rowcount == 1

    def fail_stale_running(self, timeout_seconds: int) -> int:
        """Fail stale running jobs, preserving the historical int API."""
        return len(self.fail_stale_running_jobs(timeout_seconds))

    def fail_stale_running_jobs(self, timeout_seconds: int) -> list[Dict[str, Any]]:
        """Atomically fail stale jobs and return exactly those transitions.

        Selecting stale rows and transitioning them must share one
        ``BEGIN IMMEDIATE`` transaction. A pre-read followed by a separate
        update can race a worker completion and cause cleanup code to delete
        inputs belonging to a completed job.
        """
        if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer.")
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
        ).isoformat(timespec="milliseconds")
        timestamp = _utc_now()
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            rows = connection.execute(
                """
                SELECT id, scope_id, status, payload_json, manifest_json,
                       session_id, error_code, created_at, updated_at,
                       started_at, completed_at
                FROM analysis_jobs
                WHERE status = 'running' AND started_at < ?
                ORDER BY started_at ASC
                """,
                (cutoff,),
            ).fetchall()
            if not rows:
                return []
            connection.execute(
                """
                UPDATE pilot_analyses SET failed_at = COALESCE(failed_at, ?)
                WHERE job_id IN (
                    SELECT id FROM analysis_jobs
                    WHERE status = 'running' AND started_at < ?
                )
                """,
                (timestamp, cutoff),
            )
            cursor = connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'failed', error_code = 'worker_interrupted',
                    completed_at = ?, updated_at = ?
                WHERE status = 'running' AND started_at < ?
                """,
                (timestamp, timestamp, cutoff),
            )
            if cursor.rowcount != len(rows):
                raise RuntimeError("Stale analysis job transition was incomplete.")
            # Read back under the same write transaction so callers receive
            # the terminal representation of exactly the rows transitioned.
            placeholders = ", ".join("?" for _ in rows)
            transitioned_rows = connection.execute(
                f"""
                SELECT id, scope_id, status, payload_json, manifest_json,
                       session_id, error_code, created_at, updated_at,
                       started_at, completed_at
                FROM analysis_jobs
                WHERE id IN ({placeholders})
                """,
                tuple(row["id"] for row in rows),
            ).fetchall()
        return [self._row_to_job(row) for row in transitioned_rows]

    def list_stale_running(self, timeout_seconds: int) -> list[Dict[str, Any]]:
        """Return running jobs that will be failed by the stale-job sweep."""
        if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer.")
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
        ).isoformat(timespec="milliseconds")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, scope_id, status, payload_json, manifest_json,
                       session_id, error_code, created_at, updated_at,
                       started_at, completed_at
                FROM analysis_jobs
                WHERE status = 'running' AND started_at < ?
                ORDER BY started_at ASC
                """,
                (cutoff,),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def erase_scope_if_idle(
        self,
        scope_id: str,
        pilot_scope_token: str | None = None,
        artifact_cleanup: Callable[[list[Dict[str, Any]]], Any] | None = None,
    ) -> Dict[str, int]:
        """Fence and erase one scope in a single shared-database transaction.

        The cleanup callback receives every decoded job for the scope while
        the ``BEGIN IMMEDIATE`` transaction is held.  This prevents another
        writer from creating work or changing the set of artifact references
        between the callback's file operations and the database purge.  File
        operations cannot be rolled back; a callback failure leaves all rows
        and the fence untouched so the operation can be retried safely.
        """
        normalized_scope_id = _validated_scope_id(scope_id)
        normalized_token = _validated_pilot_scope_token(pilot_scope_token)
        if artifact_cleanup is not None and not callable(artifact_cleanup):
            raise TypeError("artifact_cleanup must be callable.")

        job_columns = (
            "id, scope_id, status, payload_json, manifest_json, session_id, "
            "error_code, created_at, updated_at, started_at, completed_at"
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT COUNT(*) FROM analysis_jobs
                WHERE scope_id = ? AND status IN ('queued', 'running')
                """,
                (normalized_scope_id,),
            ).fetchone()[0]
            if active:
                raise ScopeHasActiveJobsError(
                    "This scope still has queued or running analysis jobs."
                )

            rows = connection.execute(
                f"""
                SELECT {job_columns}
                FROM analysis_jobs
                WHERE scope_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (normalized_scope_id,),
            ).fetchall()
            jobs = [self._row_to_job(row) for row in rows]
            # _row_to_job only returns None for a missing row, which cannot
            # happen for rows returned by the query.
            owned_jobs = [job for job in jobs if job is not None]

            foreign_names: set[str] = set()
            foreign_session_ids: set[str] = set()
            foreign_payloads_unknown = False
            foreign_rows = connection.execute(
                "SELECT session_id, payload_json FROM analysis_jobs WHERE scope_id != ?",
                (normalized_scope_id,),
            ).fetchall()
            for row in foreign_rows:
                raw_foreign_session_id = row["session_id"]
                if raw_foreign_session_id is not None:
                    try:
                        foreign_session_id = str(uuid.UUID(raw_foreign_session_id))
                    except (AttributeError, TypeError, ValueError):
                        foreign_payloads_unknown = True
                    else:
                        if foreign_session_id != raw_foreign_session_id:
                            foreign_payloads_unknown = True
                        else:
                            foreign_session_ids.add(foreign_session_id)
                try:
                    foreign_names.update(_upload_filenames(json.loads(row["payload_json"])))
                except (TypeError, ValueError, json.JSONDecodeError):
                    # Corrupt foreign payloads cannot safely identify a file;
                    # expose the unknown state so an artifact helper can fail
                    # closed instead of deleting an uncertain upload.
                    foreign_payloads_unknown = True
                    continue

            callback_jobs = ScopeDeletionJobs(
                owned_jobs,
                foreign_names,
                foreign_scope_payloads_unknown=foreign_payloads_unknown,
                foreign_scope_session_ids=foreign_session_ids,
            )
            if artifact_cleanup is not None:
                artifact_cleanup(callback_jobs)

            connection.execute(
                "INSERT OR IGNORE INTO deleted_scopes (scope_id) VALUES (?)",
                (normalized_scope_id,),
            )

            decision_table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'decision_cases'
                """
            ).fetchone()
            if decision_table is None:
                decision_count = 0
            else:
                decision_count = connection.execute(
                    "SELECT COUNT(*) FROM decision_cases WHERE scope_id = ?",
                    (normalized_scope_id,),
                ).fetchone()[0]

            # Use a correlated scope subquery rather than expanding every
            # historical job id into SQL parameters.  A pilot account can
            # have more rows than SQLite's variable limit (commonly 999),
            # especially when terminal-job retention is configured loosely.
            # The subquery is evaluated while this write transaction is held,
            # before the owned jobs are deleted below.
            if normalized_token is not None:
                pilot_where = (
                    "scope_token = ? OR job_id IN "
                    "(SELECT id FROM analysis_jobs WHERE scope_id = ?)"
                )
                pilot_parameters = [normalized_token, normalized_scope_id]
            else:
                pilot_where = (
                    "job_id IN "
                    "(SELECT id FROM analysis_jobs WHERE scope_id = ?)"
                )
                pilot_parameters = [normalized_scope_id]
            pilot_count = connection.execute(
                f"SELECT COUNT(*) FROM pilot_analyses WHERE {pilot_where}",
                pilot_parameters,
            ).fetchone()[0]
            connection.execute(
                f"DELETE FROM pilot_analyses WHERE {pilot_where}",
                pilot_parameters,
            )

            if decision_table is not None:
                connection.execute(
                    "DELETE FROM decision_cases WHERE scope_id = ?",
                    (normalized_scope_id,),
                )
            job_count = connection.execute(
                "DELETE FROM analysis_jobs WHERE scope_id = ?",
                (normalized_scope_id,),
            ).rowcount
        return {
            "jobs": job_count,
            "decision_cases": decision_count,
            "pilot_analyses": pilot_count,
        }

    def run_if_scope_open(
        self,
        scope_id: str,
        callback: Callable[..., Any],
    ) -> Any:
        """Run a guarded callback while the scope is open.

        The callback receives the already-locked connection.  Callers that
        need database reads must use it rather than opening a second SQLite
        connection, which would otherwise deadlock behind this transaction.
        """
        normalized_scope_id = _validated_scope_id(scope_id)
        if not callable(callback):
            raise TypeError("callback must be callable.")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM deleted_scopes WHERE scope_id = ?",
                (normalized_scope_id,),
            ).fetchone() is not None:
                raise ScopeClosedError("This scope has been closed.")
            # Keep the original no-argument callback contract for filesystem
            # writers while allowing database-aware callbacks (such as the
            # account export) to reuse this locked connection.  Bind first so
            # a TypeError raised *inside* a callback is never mistaken for a
            # signature mismatch and retried.
            try:
                inspect.signature(callback).bind(connection)
            except (TypeError, ValueError):
                return callback()
            return callback(connection)

    def purge_terminal(self, retention_hours: int) -> int:
        if not isinstance(retention_hours, int) or retention_hours < 1:
            raise ValueError("retention_hours must be a positive integer.")
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        ).isoformat(timespec="milliseconds")
        with self._connection() as connection:
            purge_metrics(connection)
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
            initialize_metrics(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS deleted_scopes (
                    scope_id TEXT PRIMARY KEY
                );

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
