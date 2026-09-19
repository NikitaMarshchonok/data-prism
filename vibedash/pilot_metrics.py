"""Opt-in pilot measurement; no dataset content or user-entered text."""

from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from pathlib import Path
import re
import sqlite3
import statistics


RETENTION_DAYS = 30
MAX_RECORDS = 10000
STAGES = frozenset({"started_at", "completed_at", "failed_at", "decision_at", "outcome_at"})
USEFULNESS = ("useful", "partly_useful", "not_useful")
BLOCKERS = ("none", "unclear_result", "missing_context", "data_quality", "missing_feature")


def scope_token(scope_id, secret):
    """Domain-separated pseudonym, not an anonymous person identifier."""
    if not isinstance(scope_id, str) or not re.fullmatch(r"[0-9a-f]{32}", scope_id):
        raise ValueError("Invalid VibeDash scope.")
    key = secret.encode() if isinstance(secret, str) else secret
    return hmac.new(key, b"pilot-metrics-v1:" + scope_id.encode(), hashlib.sha256).hexdigest()


def initialize_metrics(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS pilot_analyses (
            job_id TEXT PRIMARY KEY,
            scope_token TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('demo', 'upload')),
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            failed_at TEXT,
            decision_at TEXT,
            outcome_at TEXT,
            usefulness TEXT CHECK (usefulness IN ('useful', 'partly_useful', 'not_useful')),
            blocker TEXT CHECK (blocker IN ('none', 'unclear_result', 'missing_context', 'data_quality', 'missing_feature'))
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_pilot_created ON pilot_analyses(created_at)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_pilot_scope ON pilot_analyses(scope_token)")


def purge_metrics(connection, now=None):
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=RETENTION_DAYS)).isoformat(timespec="milliseconds")
    return connection.execute("DELETE FROM pilot_analyses WHERE created_at < ?", (cutoff,)).rowcount


def register_analysis(connection, job_id, token, source, timestamp):
    if token is None:
        return
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("Invalid pilot measurement token.")
    purge_metrics(connection)
    if connection.execute("SELECT COUNT(*) FROM pilot_analyses").fetchone()[0] >= MAX_RECORDS:
        return
    connection.execute(
        "INSERT INTO pilot_analyses (job_id, scope_token, source, created_at) VALUES (?, ?, ?, ?)",
        (job_id, token, source, timestamp),
    )


def mark_stage(connection, job_id, stage, timestamp):
    if stage not in STAGES:
        raise ValueError("Unsupported pilot stage.")
    # Only opted-in, retained records exist; a deleted record is never recreated.
    connection.execute(
        f"UPDATE pilot_analyses SET {stage} = COALESCE({stage}, ?) WHERE job_id = ?",
        (timestamp, job_id),
    )


def record_feedback(database_path, job_id, usefulness, blocker):
    if usefulness not in USEFULNESS or blocker not in BLOCKERS:
        raise ValueError("Choose one usefulness rating and one listed blocker.")
    with closing(sqlite3.connect(database_path, timeout=5)) as connection, connection:
        purge_metrics(connection)
        return connection.execute(
            "UPDATE pilot_analyses SET usefulness = ?, blocker = ? WHERE job_id = ? AND completed_at IS NOT NULL",
            (usefulness, blocker, job_id),
        ).rowcount == 1


def forget_scope(database_path, token):
    with closing(sqlite3.connect(database_path, timeout=5)) as connection, connection:
        return connection.execute("DELETE FROM pilot_analyses WHERE scope_token = ?", (token,)).rowcount


def feedback_available(database_path, job_id):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat(timespec="milliseconds")
    with closing(sqlite3.connect(database_path, timeout=5)) as connection:
        return connection.execute(
            "SELECT 1 FROM pilot_analyses WHERE job_id = ? AND created_at >= ? AND completed_at IS NOT NULL",
            (job_id, cutoff),
        ).fetchone() is not None


def _ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def _cohort(rows):
    completed = [row for row in rows if row['completed_at']]
    decisions = [row for row in completed if row['decision_at']]
    outcomes = [row for row in decisions if row['outcome_at']]
    scopes = Counter(row['scope_token'] for row in completed)
    dates = {}
    for row in completed:
        dates.setdefault(row['scope_token'], set()).add(row['created_at'][:10])
    seconds = [
        (datetime.fromisoformat(row['decision_at']) - datetime.fromisoformat(row['created_at'])).total_seconds()
        for row in decisions
    ]
    return {
        'accepted_analyses': len(rows),
        'started_analyses': sum(bool(row['started_at']) for row in rows),
        'completed_analyses': len(completed),
        'failed_analyses': sum(bool(row['failed_at']) for row in rows),
        'analyses_with_decision': len(decisions),
        'analyses_with_recorded_outcome': len(outcomes),
        'completion_rate': _ratio(len(completed), len(rows)),
        'decision_rate_among_completed': _ratio(len(decisions), len(completed)),
        'outcome_rate_among_decision_analyses': _ratio(len(outcomes), len(decisions)),
        'median_seconds_to_first_decision': round(statistics.median(seconds), 1) if seconds else None,
        'browser_scopes_with_completed_analysis': len(scopes),
        'browser_scopes_with_repeat_completed_analysis': sum(count >= 2 for count in scopes.values()),
        'browser_scopes_active_on_multiple_utc_dates': sum(len(values) >= 2 for values in dates.values()),
        'feedback_responses': sum(bool(row['usefulness']) for row in completed),
        'usefulness': {value: sum(row['usefulness'] == value for row in completed) for value in USEFULNESS},
        'blockers': {value: sum(row['blocker'] == value for row in completed) for value in BLOCKERS},
    }


def build_pilot_report(database_path, days=30, now=None):
    """Read-only aggregate report; never returns tokens, IDs, or row contents."""
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= RETENTION_DAYS:
        raise ValueError(f"days must be between 1 and {RETENTION_DAYS}.")
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).isoformat(timespec="milliseconds")
    path = Path(database_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        installed = connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'pilot_analyses' AND type = 'table'").fetchone()
        rows = connection.execute(
            "SELECT * FROM pilot_analyses WHERE created_at >= ? AND created_at <= ?",
            (cutoff, now.isoformat(timespec="milliseconds")),
        ).fetchall() if installed else []
        stored = connection.execute("SELECT COUNT(*) FROM pilot_analyses").fetchone()[0] if installed else 0
    return {
        'contract': 'pilot-funnel-v1',
        'generated_at': now.isoformat(timespec="seconds"),
        'cohort_days': days,
        'collection_installed': bool(installed),
        'record_limit_reached': stored >= MAX_RECORDS,
        'cohorts': {source: _cohort([row for row in rows if row['source'] == source]) for source in ('upload', 'demo')},
        'limitations': [
            'Opt-in accepted background analyses only; preflight rejections and synchronous previews are excluded.',
            'Cohorts use analysis creation time. Recent cases may not be due for review yet.',
            'One analysis counts once per stage, regardless of polling, number of cases, or repeated outcome edits.',
            'Scope tokens are pseudonymous, not people or companies; clearing cookies changes a guest scope, while a pilot account can restore its account scope by signing in again.',
            'Outcomes are self-reported; recorded outcomes do not prove causal impact or willingness to pay.',
            'Thirty-day retention, record limits, consent withdrawal, and ephemeral storage can reduce coverage.',
        ],
    }
