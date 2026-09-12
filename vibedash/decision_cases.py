"""Persistent, session-scoped decision cases created from analysis evidence."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator


IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{32}$")
TERMINAL_STATUSES = frozenset({"validated", "invalidated", "cancelled"})
ALLOWED_STATUSES = frozenset({"tracking", *TERMINAL_STATUSES})
DEFAULT_DECISION_CASES_PER_SCOPE = 50
MAX_DECISION_CASES = 500


class DecisionCaseCapacityError(RuntimeError):
    """A browser scope has reached the bounded decision-case limit."""


class DecisionCaseConflictError(RuntimeError):
    """The same analysis priority is already tracked as a decision case."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _validated_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {label} identifier.")
    return value


def _bounded_text(
    value: Any,
    label: str,
    *,
    maximum: int,
    required: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    normalized = " ".join(value.split())
    if required and not normalized:
        raise ValueError(f"{label} is required.")
    if len(normalized) > maximum:
        raise ValueError(f"{label} must not exceed {maximum} characters.")
    return normalized


def _review_date(value: Any) -> str:
    normalized = _bounded_text(value, "Review date", maximum=10)
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("Review date must use YYYY-MM-DD format.") from error
    return parsed.isoformat()


def _serialized_snapshot(value: Dict[str, Any]) -> str:
    if not isinstance(value, dict):
        raise ValueError("Evidence snapshot must be an object.")
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if len(serialized.encode("utf-8")) > 16 * 1024:
        raise ValueError("Evidence snapshot exceeds the storage limit.")
    return serialized


class DecisionCaseStore:
    """Store a bounded evidence-to-outcome workflow without source rows."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(
        self,
        scope_id: str,
        job_id: str,
        *,
        priority: int,
        owner: str,
        decision: str,
        success_metric: str,
        target_outcome: str,
        review_date: str,
        evidence_snapshot: Dict[str, Any],
        max_cases_per_scope: int = DEFAULT_DECISION_CASES_PER_SCOPE,
    ) -> Dict[str, Any]:
        normalized_scope = _validated_identifier(scope_id, "scope")
        normalized_job = _validated_identifier(job_id, "analysis job")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("Priority must be an integer.")
        if not 1 <= priority <= 3:
            raise ValueError("Priority must be between 1 and 3.")
        if (
            isinstance(max_cases_per_scope, bool)
            or not isinstance(max_cases_per_scope, int)
            or max_cases_per_scope < 1
        ):
            raise ValueError("max_cases_per_scope must be a positive integer.")

        values = {
            "owner": _bounded_text(owner, "Owner", maximum=120),
            "decision": _bounded_text(decision, "Decision", maximum=1000),
            "success_metric": _bounded_text(
                success_metric,
                "Success metric",
                maximum=240,
            ),
            "target_outcome": _bounded_text(
                target_outcome,
                "Target outcome",
                maximum=500,
            ),
            "review_date": _review_date(review_date),
            "snapshot_json": _serialized_snapshot(evidence_snapshot),
        }
        case_id = uuid.uuid4().hex
        timestamp = _utc_now()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing_count = connection.execute(
                    "SELECT COUNT(*) FROM decision_cases WHERE scope_id = ?",
                    (normalized_scope,),
                ).fetchone()[0]
                if existing_count >= max_cases_per_scope:
                    raise DecisionCaseCapacityError(
                        "The decision-case limit has been reached."
                    )
                connection.execute(
                    """
                    INSERT INTO decision_cases (
                        id, scope_id, job_id, priority, status, owner,
                        decision, success_metric, target_outcome, review_date,
                        evidence_snapshot_json, actual_outcome,
                        created_at, updated_at, resolved_at
                    ) VALUES (?, ?, ?, ?, 'tracking', ?, ?, ?, ?, ?, ?, '', ?, ?, NULL)
                    """,
                    (
                        case_id,
                        normalized_scope,
                        normalized_job,
                        priority,
                        values["owner"],
                        values["decision"],
                        values["success_metric"],
                        values["target_outcome"],
                        values["review_date"],
                        values["snapshot_json"],
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise DecisionCaseConflictError(
                "This analysis priority is already being tracked."
            ) from error
        return self.get(case_id, normalized_scope)

    def get(
        self,
        case_id: str,
        scope_id: str | None = None,
    ) -> Dict[str, Any] | None:
        normalized_id = _validated_identifier(case_id, "decision case")
        scope_filter = ""
        parameters: tuple[Any, ...] = (normalized_id,)
        if scope_id is not None:
            normalized_scope = _validated_identifier(scope_id, "scope")
            scope_filter = " AND scope_id = ?"
            parameters = (normalized_id, normalized_scope)
        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT id, scope_id, job_id, priority, status, owner,
                       decision, success_metric, target_outcome, review_date,
                       evidence_snapshot_json, actual_outcome,
                       created_at, updated_at, resolved_at
                FROM decision_cases
                WHERE id = ?{scope_filter}
                """,
                parameters,
            ).fetchone()
        return self._row_to_case(row)

    def find_for_job_priority(
        self,
        scope_id: str,
        job_id: str,
        priority: int,
    ) -> Dict[str, Any] | None:
        normalized_scope = _validated_identifier(scope_id, "scope")
        normalized_job = _validated_identifier(job_id, "analysis job")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, scope_id, job_id, priority, status, owner,
                       decision, success_metric, target_outcome, review_date,
                       evidence_snapshot_json, actual_outcome,
                       created_at, updated_at, resolved_at
                FROM decision_cases
                WHERE scope_id = ? AND job_id = ? AND priority = ?
                """,
                (normalized_scope, normalized_job, priority),
            ).fetchone()
        return self._row_to_case(row)

    def list_for_scope(
        self,
        scope_id: str,
        *,
        limit: int = MAX_DECISION_CASES,
    ) -> list[Dict[str, Any]]:
        normalized_scope = _validated_identifier(scope_id, "scope")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer.")
        if not 1 <= limit <= MAX_DECISION_CASES:
            raise ValueError(f"limit must be between 1 and {MAX_DECISION_CASES}.")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, scope_id, job_id, priority, status, owner,
                       decision, success_metric, target_outcome, review_date,
                       evidence_snapshot_json, actual_outcome,
                       created_at, updated_at, resolved_at
                FROM decision_cases
                WHERE scope_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (normalized_scope, limit),
            ).fetchall()
        return [self._row_to_case(row) for row in rows]

    def update_outcome(
        self,
        case_id: str,
        scope_id: str,
        *,
        status: str,
        actual_outcome: str,
    ) -> Dict[str, Any] | None:
        normalized_id = _validated_identifier(case_id, "decision case")
        normalized_scope = _validated_identifier(scope_id, "scope")
        if status not in ALLOWED_STATUSES:
            raise ValueError("Unsupported decision-case status.")
        outcome = _bounded_text(
            actual_outcome,
            "Actual outcome",
            maximum=1200,
            required=status in TERMINAL_STATUSES,
        )
        timestamp = _utc_now()
        resolved_at = timestamp if status in TERMINAL_STATUSES else None
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE decision_cases
                SET status = ?, actual_outcome = ?, updated_at = ?, resolved_at = ?
                WHERE id = ? AND scope_id = ?
                """,
                (
                    status,
                    outcome,
                    timestamp,
                    resolved_at,
                    normalized_id,
                    normalized_scope,
                ),
            )
        if cursor.rowcount != 1:
            return None
        return self.get(normalized_id, normalized_scope)

    def purge_terminal(self, retention_days: int) -> int:
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, int)
            or retention_days < 1
        ):
            raise ValueError("retention_days must be a positive integer.")
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat(timespec="milliseconds")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM decision_cases
                WHERE status IN ('validated', 'invalidated', 'cancelled')
                  AND updated_at < ?
                """,
                (cutoff,),
            )
        return cursor.rowcount

    @staticmethod
    def _row_to_case(row: sqlite3.Row | None) -> Dict[str, Any] | None:
        if row is None:
            return None
        decision_case = dict(row)
        decision_case["evidence_snapshot"] = json.loads(
            decision_case.pop("evidence_snapshot_json")
        )
        return decision_case

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
                CREATE TABLE IF NOT EXISTS decision_cases (
                    id TEXT PRIMARY KEY,
                    scope_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    priority INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 3),
                    status TEXT NOT NULL CHECK (
                        status IN ('tracking', 'validated', 'invalidated', 'cancelled')
                    ),
                    owner TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    success_metric TEXT NOT NULL,
                    target_outcome TEXT NOT NULL,
                    review_date TEXT NOT NULL,
                    evidence_snapshot_json TEXT NOT NULL,
                    actual_outcome TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT,
                    UNIQUE(scope_id, job_id, priority)
                );

                CREATE INDEX IF NOT EXISTS idx_decision_cases_scope_time
                ON decision_cases(scope_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_decision_cases_status_review
                ON decision_cases(status, review_date ASC);
                """
            )
