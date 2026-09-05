"""SQLite persistence for scoped data-drift runs and in-app alerts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List


VALID_STATUSES = {"stable", "warning", "critical"}
ALERT_STATUSES = {"warning", "critical"}
DEFAULT_RETENTION = 100
MAX_QUERY_LIMIT = 100


class DriftStore:
    """Persist drift reports for one isolated application scope."""

    def __init__(
        self,
        database_path: str | Path,
        scope_id: str,
        *,
        retention: int = DEFAULT_RETENTION,
    ) -> None:
        if not isinstance(scope_id, str) or not scope_id or len(scope_id) > 128:
            raise ValueError("scope_id must be a non-empty string of at most 128 characters.")
        if not isinstance(retention, int) or retention < 1:
            raise ValueError("retention must be a positive integer.")
        self.database_path = Path(database_path)
        self.scope_id = scope_id
        self.retention = retention
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def record_run(
        self,
        report: Dict[str, Any],
        *,
        batch_id: str,
        dataset_name: str | None = None,
        checked_at: str | None = None,
    ) -> Dict[str, Any]:
        """Store a report once per baseline and batch, creating an alert if needed."""
        status = report.get("status") if isinstance(report, dict) else None
        if status not in VALID_STATUSES:
            raise ValueError("Drift report has an unsupported status.")
        if not isinstance(batch_id, str) or not batch_id or len(batch_id) > 255:
            raise ValueError("batch_id must be a non-empty string of at most 255 characters.")
        baseline_created_at = report.get("baseline_created_at")
        if not isinstance(baseline_created_at, str) or not baseline_created_at:
            raise ValueError("Drift report is missing baseline_created_at.")

        timestamp = checked_at or _utc_now()
        report_json = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        run_id = uuid.uuid4().hex

        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO drift_runs (
                    id, scope_id, batch_id, dataset_name, baseline_created_at,
                    checked_at, status, summary, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self.scope_id,
                    batch_id,
                    dataset_name,
                    baseline_created_at,
                    timestamp,
                    status,
                    str(report.get("summary", "")),
                    report_json,
                ),
            )
            created = cursor.rowcount == 1
            if not created:
                existing = connection.execute(
                    """
                    SELECT id, checked_at FROM drift_runs
                    WHERE scope_id = ? AND batch_id = ? AND baseline_created_at = ?
                    """,
                    (self.scope_id, batch_id, baseline_created_at),
                ).fetchone()
                return {
                    "run_id": existing["id"],
                    "checked_at": existing["checked_at"],
                    "status": status,
                    "created": False,
                    "alert_created": False,
                }

            alert_created = False
            if status in ALERT_STATUSES:
                connection.execute(
                    """
                    INSERT INTO drift_alerts (
                        run_id, scope_id, created_at, severity, title, message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        self.scope_id,
                        timestamp,
                        status,
                        _alert_title(status),
                        str(report.get("summary", "")),
                    ),
                )
                alert_created = True
            self._enforce_retention(connection)

        return {
            "run_id": run_id,
            "checked_at": timestamp,
            "status": status,
            "created": True,
            "alert_created": alert_created,
        }

    def list_runs(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent compact run summaries for this scope."""
        query_limit = _validated_limit(limit)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, batch_id, dataset_name, baseline_created_at,
                       checked_at, status, summary
                FROM drift_runs
                WHERE scope_id = ?
                ORDER BY checked_at DESC, rowid DESC
                LIMIT ?
                """,
                (self.scope_id, query_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> Dict[str, Any] | None:
        """Return one full report only when it belongs to this scope."""
        if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
            raise ValueError("run_id must be a non-empty string of at most 128 characters.")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, batch_id, dataset_name, baseline_created_at,
                       checked_at, status, summary, report_json
                FROM drift_runs
                WHERE id = ? AND scope_id = ?
                """,
                (run_id, self.scope_id),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["report"] = json.loads(result.pop("report_json"))
        return result

    def list_alerts(
        self,
        *,
        unacknowledged_only: bool = True,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return recent alerts for this scope."""
        query_limit = _validated_limit(limit)
        acknowledgement_filter = "AND acknowledged_at IS NULL" if unacknowledged_only else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id, run_id, created_at, severity, title, message,
                       acknowledged_at
                FROM drift_alerts
                WHERE scope_id = ? {acknowledgement_filter}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (self.scope_id, query_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def acknowledge_alert(self, alert_id: int, *, acknowledged_at: str | None = None) -> bool:
        """Acknowledge one alert only when it belongs to this scope."""
        if not isinstance(alert_id, int) or alert_id < 1:
            raise ValueError("alert_id must be a positive integer.")
        timestamp = acknowledged_at or _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE drift_alerts
                SET acknowledged_at = ?
                WHERE id = ? AND scope_id = ? AND acknowledged_at IS NULL
                """,
                (timestamp, alert_id, self.scope_id),
            )
        return cursor.rowcount == 1

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
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
                CREATE TABLE IF NOT EXISTS drift_runs (
                    id TEXT PRIMARY KEY,
                    scope_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    dataset_name TEXT,
                    baseline_created_at TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    UNIQUE(scope_id, batch_id, baseline_created_at)
                );

                CREATE INDEX IF NOT EXISTS idx_drift_runs_scope_time
                ON drift_runs(scope_id, checked_at DESC);

                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    acknowledged_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES drift_runs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_drift_alerts_scope_time
                ON drift_alerts(scope_id, created_at DESC);
                """
            )

    def _enforce_retention(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM drift_runs
            WHERE scope_id = ? AND id NOT IN (
                SELECT id FROM drift_runs
                WHERE scope_id = ?
                ORDER BY checked_at DESC, rowid DESC
                LIMIT ?
            )
            """,
            (self.scope_id, self.scope_id, self.retention),
        )


def _validated_limit(limit: int) -> int:
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    return min(limit, MAX_QUERY_LIMIT)


def _alert_title(status: str) -> str:
    return "Critical data drift detected" if status == "critical" else "Data drift warning"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
