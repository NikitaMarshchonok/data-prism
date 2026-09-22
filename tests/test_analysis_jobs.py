import sqlite3
import threading
import time
import unittest
import uuid
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask

from vibedash.analysis_jobs import (
    AnalysisJobCapacityError,
    AnalysisJobDispatcher,
    AnalysisJobStore,
    MAX_HISTORY_JOBS,
    ScopeClosedError,
    ScopeHasActiveJobsError,
)
from vibedash.decision_cases import DecisionCaseStore


class AnalysisJobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "analysis_jobs.sqlite3"
        )
        self.store = AnalysisJobStore(self.database_path)
        self.scope_id = uuid.uuid4().hex

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_job_is_scoped_and_payload_round_trips(self):
        job = self.store.create(
            self.scope_id,
            {"stored_filename": f"vibedash-{'a' * 32}.csv"},
        )

        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["payload"]["stored_filename"], f"vibedash-{'a' * 32}.csv")
        self.assertIsNone(self.store.get(job["id"], uuid.uuid4().hex))

    def test_claim_and_completion_are_single_transition_operations(self):
        job = self.store.create(self.scope_id, {"prompt": "Analyze revenue"})

        claimed = self.store.claim(job["id"])
        self.assertEqual(claimed["status"], "running")
        self.assertIsNone(self.store.claim(job["id"]))

        session_id = str(uuid.uuid4())
        self.assertTrue(self.store.complete(job["id"], session_id))
        self.assertFalse(self.store.complete(job["id"], str(uuid.uuid4())))
        completed = self.store.get(job["id"], self.scope_id)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["session_id"], session_id)

    def test_failure_exposes_only_a_bounded_error_code(self):
        job = self.store.create(self.scope_id, {})

        self.assertTrue(self.store.fail(job["id"], "invalid_dataset"))
        failed = self.store.get(job["id"], self.scope_id)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_code"], "invalid_dataset")
        with self.assertRaises(ValueError):
            self.store.fail(job["id"], "raw exception: secret/path")

    def test_stale_running_jobs_fail_and_expired_terminal_jobs_are_purged(self):
        job = self.store.create(self.scope_id, {})
        self.store.claim(job["id"])
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET started_at = ?, updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00.000+00:00", "2000-01-01T00:00:00.000+00:00", job["id"]),
            )

        self.assertEqual(self.store.fail_stale_running(60), 1)
        self.assertEqual(self.store.get(job["id"])["error_code"], "worker_interrupted")
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET updated_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00.000+00:00", job["id"]),
            )
        self.assertEqual(self.store.purge_terminal(24), 1)
        self.assertIsNone(self.store.get(job["id"]))

    def test_stale_transition_returns_only_jobs_failed_in_same_transaction(self):
        job = self.store.create(
            self.scope_id,
            {
                "analysis_kind": "period_comparison",
                "baseline": {"stored_filename": "vibedash-" + "a" * 32 + ".csv"},
            },
        )
        self.store.claim(job["id"])
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET started_at = ? WHERE id = ?",
                ("2000-01-01T00:00:00.000+00:00", job["id"]),
            )

        transitioned = self.store.fail_stale_running_jobs(60)

        self.assertEqual([record["id"] for record in transitioned], [job["id"]])
        self.assertEqual(transitioned[0]["status"], "failed")
        self.assertEqual(self.store.fail_stale_running(60), 0)

    def test_invalid_identifiers_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.get("../outside")
        with self.assertRaises(ValueError):
            self.store.create("not-a-scope", {})

    def test_active_job_capacity_is_enforced_atomically(self):
        self.store.create(
            self.scope_id,
            {},
            max_active_per_scope=1,
            max_active_total=10,
        )

        with self.assertRaises(AnalysisJobCapacityError):
            self.store.create(
                self.scope_id,
                {},
                max_active_per_scope=1,
                max_active_total=10,
            )
        with self.assertRaises(AnalysisJobCapacityError):
            self.store.create(
                uuid.uuid4().hex,
                {},
                max_active_per_scope=10,
                max_active_total=1,
            )

    def test_scope_erase_rejects_active_jobs_without_mutation(self):
        job = self.store.create(self.scope_id, {})

        with self.assertRaises(ScopeHasActiveJobsError):
            self.store.erase_scope_if_idle(self.scope_id)

        self.assertIsNotNone(self.store.get(job["id"], self.scope_id))
        with sqlite3.connect(self.database_path) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM deleted_scopes WHERE scope_id = ?",
                    (self.scope_id,),
                ).fetchone()
            )

    def test_scope_erase_rolls_back_callback_failure_then_is_retryable(self):
        pilot_token = "a" * 64
        job = self.store.create(
            self.scope_id,
            {"stored_filename": f"vibedash-{'a' * 32}.csv"},
            pilot_scope_token=pilot_token,
        )
        self.store.fail(job["id"])
        calls = []

        def fail_cleanup(jobs):
            calls.append(list(jobs))
            raise RuntimeError("cleanup failed")

        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.store.erase_scope_if_idle(
                self.scope_id,
                pilot_token,
                fail_cleanup,
            )
        self.assertIsNotNone(self.store.get(job["id"], self.scope_id))

        result = self.store.erase_scope_if_idle(
            self.scope_id,
            pilot_token,
            lambda jobs: calls.append(list(jobs)),
        )
        self.assertEqual(result, {"jobs": 1, "decision_cases": 0, "pilot_analyses": 1})
        self.assertEqual([item["id"] for item in calls[-1]], [job["id"]])
        with self.assertRaises(ScopeClosedError):
            self.store.create(self.scope_id, {})

        # A fenced scope can safely be retried after partial file cleanup.
        self.assertEqual(
            self.store.erase_scope_if_idle(self.scope_id),
            {"jobs": 0, "decision_cases": 0, "pilot_analyses": 0},
        )

    def test_scope_erase_preserves_other_scope_and_exposes_foreign_uploads(self):
        other_scope = uuid.uuid4().hex
        shared_name = f"vibedash-{'b' * 32}.csv"
        owned = self.store.create(
            self.scope_id,
            {"stored_filename": shared_name},
        )
        other = self.store.create(
            other_scope,
            {
                "analysis_kind": "period_comparison",
                "baseline": {"stored_filename": shared_name},
                "current": {"stored_filename": "../unsafe.csv"},
            },
        )
        self.store.fail(owned["id"])
        self.store.fail(other["id"])
        received = []

        result = self.store.erase_scope_if_idle(
            self.scope_id,
            artifact_cleanup=lambda jobs: received.append(jobs),
        )

        self.assertEqual(result["jobs"], 1)
        self.assertEqual([job["id"] for job in received[0]], [owned["id"]])
        self.assertEqual(received[0].foreign_scope_upload_filenames, {shared_name})
        self.assertFalse(received[0].foreign_scope_payloads_unknown)
        self.assertEqual(received[0].foreign_scope_session_ids, frozenset())
        self.assertIsNotNone(self.store.get(other["id"], other_scope))

    def test_scope_erase_exposes_foreign_session_references(self):
        other_scope = uuid.uuid4().hex
        foreign_session = str(uuid.uuid4())
        foreign_job = self.store.create(other_scope, {})
        self.store.claim(foreign_job["id"])
        self.store.complete(foreign_job["id"], foreign_session)
        received = []

        self.store.erase_scope_if_idle(
            self.scope_id,
            artifact_cleanup=lambda jobs: received.append(jobs),
        )
        self.assertEqual(
            received[0].foreign_scope_session_ids,
            frozenset({foreign_session}),
        )

    def test_scope_erase_fails_closed_on_noncanonical_foreign_session_reference(self):
        other_scope = uuid.uuid4().hex
        foreign_job = self.store.create(other_scope, {})
        self.store.claim(foreign_job["id"])
        foreign_session = str(uuid.uuid4())
        self.store.complete(foreign_job["id"], foreign_session)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET session_id = ? WHERE id = ?",
                (foreign_session.upper(), foreign_job["id"]),
            )
        received = []

        self.store.erase_scope_if_idle(
            self.scope_id,
            artifact_cleanup=lambda jobs: received.append(jobs),
        )
        self.assertTrue(received[0].foreign_scope_payloads_unknown)

    def test_run_if_scope_open_guards_open_and_fenced_scopes(self):
        self.assertEqual(
            self.store.run_if_scope_open(self.scope_id, lambda _connection: "ok"),
            "ok",
        )
        job = self.store.create(self.scope_id, {})
        self.store.fail(job["id"])
        self.store.erase_scope_if_idle(self.scope_id)

        with self.assertRaises(ScopeClosedError):
            self.store.run_if_scope_open(
                self.scope_id,
                lambda _connection: "not-written",
            )

    def test_scope_erase_purges_pilot_rows_without_sqlite_parameter_explosion(self):
        """Long-lived scopes must not exceed SQLite's bound-variable limit."""
        token = "a" * 64
        rows = []
        metrics = []
        for index in range(1100):
            job_id = f"{index:032x}"
            rows.append(
                (
                    job_id,
                    self.scope_id,
                    "failed",
                    json.dumps({}),
                    f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}.000+00:00",
                    f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}.000+00:00",
                )
            )
            metrics.append(
                (
                    job_id,
                    token,
                    "upload",
                    "2026-01-01T00:00:00.000+00:00",
                )
            )
        with sqlite3.connect(self.database_path) as connection:
            connection.executemany(
                """
                INSERT INTO analysis_jobs
                    (id, scope_id, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.executemany(
                """
                INSERT INTO pilot_analyses (job_id, scope_token, source, created_at)
                VALUES (?, ?, ?, ?)
                """,
                metrics,
            )

        result = self.store.erase_scope_if_idle(self.scope_id, token)

        self.assertEqual(result["jobs"], 1100)
        self.assertEqual(result["pilot_analyses"], 1100)
        with sqlite3.connect(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM analysis_jobs WHERE scope_id = ?",
                    (self.scope_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM pilot_analyses WHERE scope_token = ?",
                    (token,),
                ).fetchone()[0],
                0,
            )

    def test_manifest_and_recent_history_are_scoped(self):
        older = self.store.create(self.scope_id, {"prompt": "Older"})
        newer = self.store.create(self.scope_id, {"prompt": "Newer"})
        other = self.store.create(uuid.uuid4().hex, {"prompt": "Other"})
        self.store.claim(older["id"])
        self.store.complete(
            older["id"],
            str(uuid.uuid4()),
            {"manifest_version": 1, "dataset": {"content_sha256": "a" * 64}},
        )

        history = self.store.list_for_scope(self.scope_id)

        self.assertEqual([job["id"] for job in history], [newer["id"], older["id"]])
        self.assertEqual(history[1]["manifest"]["manifest_version"], 1)
        self.assertNotIn(other["id"], [job["id"] for job in history])
        with self.assertRaises(ValueError):
            self.store.list_for_scope(self.scope_id, limit=0)

    def test_paginated_history_is_scoped_ordered_and_reports_sentinel(self):
        older = self.store.create(self.scope_id, {"prompt": "Older"})
        newer = self.store.create(self.scope_id, {"prompt": "Newer"})
        other = self.store.create(uuid.uuid4().hex, {"prompt": "Other"})
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET created_at = ? WHERE id IN (?, ?, ?)",
                (
                    "2026-01-01T00:00:00.000+00:00",
                    older["id"],
                    newer["id"],
                    other["id"],
                ),
            )

        first_page, has_more = self.store.list_for_scope_page(
            self.scope_id,
            limit=1,
        )
        self.assertEqual([job["id"] for job in first_page], [newer["id"]])
        self.assertTrue(has_more)

        final_page, has_more = self.store.list_for_scope_page(
            self.scope_id,
            limit=2,
        )
        self.assertEqual(
            [job["id"] for job in final_page],
            [newer["id"], older["id"]],
        )
        self.assertFalse(has_more)
        self.assertNotIn(other["id"], [job["id"] for job in final_page])

    def test_paginated_history_validates_limit_boundaries(self):
        self.assertEqual(
            self.store.list_for_scope_page(
                self.scope_id,
                limit=MAX_HISTORY_JOBS,
            ),
            ([], False),
        )
        for invalid_limit in (0, MAX_HISTORY_JOBS + 1, True, 1.5):
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(ValueError):
                    self.store.list_for_scope_page(
                        self.scope_id,
                        limit=invalid_limit,
                    )
        with self.assertRaises(ValueError):
            self.store.list_for_scope_page("not-a-scope", limit=1)

    def test_existing_job_database_is_migrated_for_manifests(self):
        legacy_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute(
                """
                CREATE TABLE analysis_jobs (
                    id TEXT PRIMARY KEY,
                    scope_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    session_id TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
        migrated_store = AnalysisJobStore(legacy_path)
        job = migrated_store.create(self.scope_id, {"prompt": "Migrated"})

        self.assertIsNone(job["manifest"])
        with sqlite3.connect(legacy_path) as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(analysis_jobs)"
                ).fetchall()
            }
        self.assertIn("manifest_json", columns)


class AnalysisJobDispatcherTests(unittest.TestCase):
    def test_dispatcher_completes_a_claimed_job(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "jobs.sqlite3"
            store = AnalysisJobStore(database_path)
            job = store.create(uuid.uuid4().hex, {"prompt": "Analyze"})
            started = threading.Event()
            release = threading.Event()
            session_id = str(uuid.uuid4())
            app = Flask(__name__)
            app.config["VIBEDASH_JOB_STORE_PATH"] = str(database_path)
            dispatcher = AnalysisJobDispatcher(max_workers=1)

            def processor(_job):
                started.set()
                release.wait(timeout=2)
                return session_id

            self.assertTrue(dispatcher.submit(app, job["id"], processor))
            self.assertTrue(started.wait(timeout=2))
            self.assertFalse(dispatcher.submit(app, job["id"], processor))
            release.set()
            for _ in range(100):
                current = store.get(job["id"])
                if current["status"] == "completed":
                    break
                time.sleep(0.01)

            self.assertEqual(current["status"], "completed")
            self.assertEqual(current["session_id"], session_id)

    def test_dispatcher_records_a_sanitized_failure(self):
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "jobs.sqlite3"
            store = AnalysisJobStore(database_path)
            job = store.create(uuid.uuid4().hex, {})
            app = Flask(__name__)
            app.config["VIBEDASH_JOB_STORE_PATH"] = str(database_path)
            app.logger.disabled = True
            dispatcher = AnalysisJobDispatcher(max_workers=1)

            def processor(_job):
                raise RuntimeError("sensitive internal details")

            dispatcher.submit(app, job["id"], processor)
            for _ in range(100):
                current = store.get(job["id"])
                if current["status"] == "failed":
                    break
                time.sleep(0.01)

            self.assertEqual(current["status"], "failed")
            self.assertEqual(current["error_code"], "analysis_failed")
            self.assertNotIn("sensitive", str(current))


if __name__ == "__main__":
    unittest.main()
