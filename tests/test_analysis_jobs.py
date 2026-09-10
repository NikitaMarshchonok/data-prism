import sqlite3
import threading
import time
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask

from vibedash.analysis_jobs import (
    AnalysisJobCapacityError,
    AnalysisJobDispatcher,
    AnalysisJobStore,
)


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
