import os
import time
import unittest
import uuid
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app

from vibedash.analysis_jobs import AnalysisJobStore


class VibeDashJobRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.previous_config = {
            key: web_app.app.config[key]
            for key in (
                "UPLOAD_FOLDER",
                "VIBEDASH_JOB_STORE_PATH",
                "VIBEDASH_RETENTION_HOURS",
                "VIBEDASH_JOB_TIMEOUT_SECONDS",
                "VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE",
                "VIBEDASH_MAX_ACTIVE_JOBS",
            )
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(root / "uploads"),
            VIBEDASH_JOB_STORE_PATH=str(root / "jobs.sqlite3"),
            VIBEDASH_RETENTION_HOURS=24,
            VIBEDASH_JOB_TIMEOUT_SECONDS=600,
            VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE=2,
            VIBEDASH_MAX_ACTIVE_JOBS=25,
        )
        Path(web_app.app.config["UPLOAD_FOLDER"]).mkdir()

    def tearDown(self):
        web_app.app.config.update(self.previous_config)
        self.temporary_directory.cleanup()

    @patch("vibedash.routes.analysis_job_dispatcher.submit", return_value=True)
    def test_demo_job_is_queued_and_visible_only_to_its_session(self, submit):
        with web_app.app.test_client() as owner:
            response = owner.post(
                "/vibedash/jobs",
                data={
                    "demo_dataset": "saas_growth",
                    "prompt": "Show evidence-backed SaaS performance.",
                },
            )
            payload = response.get_json()
            submit.assert_called_once()
            submit.reset_mock()
            status_response = owner.get(payload["status_url"])

        self.assertEqual(response.status_code, 202)
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.get_json()["status"], "queued")
        submit.assert_called_once()
        self.assertEqual(
            len(list(Path(web_app.app.config["UPLOAD_FOLDER"]).glob("vibedash-*.csv"))),
            1,
        )

        with web_app.app.test_client() as stranger:
            unauthorized = stranger.get(payload["status_url"])
        self.assertEqual(unauthorized.status_code, 404)

    @patch("vibedash.routes.analysis_job_dispatcher.submit", return_value=True)
    def test_invalid_job_requests_do_not_leave_uploads(self, _submit):
        with web_app.app.test_client() as client:
            response = client.post(
                "/vibedash/jobs",
                data={"demo_dataset": "unknown", "prompt": "Analyze"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(list(Path(web_app.app.config["UPLOAD_FOLDER"]).iterdir()), [])

    def test_preflight_blocks_duplicate_csv_headers(self):
        rows = "\n".join(f"{index},{index + 1}" for index in range(40))
        duplicate_header_csv = f"value,value\n{rows}\n".encode()

        with web_app.app.test_client() as client:
            response = client.post(
                "/vibedash/readiness",
                data={
                    "datafile": (BytesIO(duplicate_header_csv), "duplicates.csv")
                },
            )

        report = response.get_json()["readiness"]
        identity = next(
            check
            for check in report["checks"]
            if check["check_id"] == "column-identity"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(identity["status"], "blocked")

    def test_readiness_preflight_does_not_retain_uploaded_rows(self):
        csv_rows = "date,revenue,region\n" + "\n".join(
            f"2026-01-{(index % 28) + 1:02d},{index * 10},{'north' if index % 2 else 'south'}"
            for index in range(40)
        )
        with web_app.app.test_client() as client:
            response = client.post(
                "/vibedash/readiness",
                data={"datafile": (BytesIO(csv_rows.encode()), "clean.csv")},
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["readiness"]["analysis_allowed"])
        self.assertEqual(payload["readiness"]["contract"], "dataset-readiness-v1")
        self.assertEqual(list(Path(web_app.app.config["UPLOAD_FOLDER"]).iterdir()), [])

    @patch("vibedash.routes.analysis_job_dispatcher.submit", return_value=True)
    def test_blocked_dataset_is_not_queued_or_retained(self, submit):
        with web_app.app.test_client() as client:
            response = client.post(
                "/vibedash/jobs",
                data={
                    "prompt": "Analyze this table",
                    "datafile": (BytesIO(b"constant\n1\n1\n1\n1\n1\n"), "blocked.csv"),
                },
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 422)
        self.assertEqual(payload["readiness"]["status"], "blocked")
        self.assertFalse(payload["readiness"]["analysis_allowed"])
        submit.assert_not_called()
        self.assertEqual(list(Path(web_app.app.config["UPLOAD_FOLDER"]).iterdir()), [])

    @patch("vibedash.routes.analysis_job_dispatcher.submit", return_value=True)
    def test_session_job_limit_returns_429_and_removes_rejected_upload(self, _submit):
        with web_app.app.test_client() as client:
            first = client.post(
                "/vibedash/jobs",
                data={"demo_dataset": "saas_growth", "prompt": "First"},
            )
            second = client.post(
                "/vibedash/jobs",
                data={"demo_dataset": "saas_growth", "prompt": "Second"},
            )
            rejected = client.post(
                "/vibedash/jobs",
                data={"demo_dataset": "saas_growth", "prompt": "Third"},
            )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(
            len(list(Path(web_app.app.config["UPLOAD_FOLDER"]).glob("vibedash-*.csv"))),
            2,
        )

    @patch("vibedash.routes.analysis_job_dispatcher.submit", return_value=True)
    def test_failed_job_status_does_not_expose_internal_details(self, _submit):
        with web_app.app.test_client() as client:
            queued = client.post(
                "/vibedash/jobs",
                data={"demo_dataset": "saas_growth", "prompt": "Analyze"},
            ).get_json()
            store = AnalysisJobStore(web_app.app.config["VIBEDASH_JOB_STORE_PATH"])
            store.fail(queued["job_id"], "invalid_dataset")

            response = client.get(queued["status_url"])

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "The analysis could not be completed.")
        self.assertNotIn("invalid_dataset", str(payload))

    @patch("vibedash.routes.analysis_job_dispatcher.submit", return_value=True)
    @patch("vibedash.routes.load_session_data")
    def test_completed_job_result_is_rendered_for_owner(self, load_session, _submit):
        load_session.return_value = {
            "viz_spec": {"title": "Completed evidence dashboard"},
            "dashboard_data": {
                "kpis": [],
                "charts": [],
                "tables": [],
                "insights": [],
                "statistical_validation": {"tests": []},
                "pattern_analysis": {"status": "insufficient_data"},
            },
            "filename": "demo.csv",
            "prompt": "Analyze",
        }
        with web_app.app.test_client() as owner:
            queued = owner.post(
                "/vibedash/jobs",
                data={"demo_dataset": "saas_growth", "prompt": "Analyze"},
            ).get_json()
            store = AnalysisJobStore(web_app.app.config["VIBEDASH_JOB_STORE_PATH"])
            store.claim(queued["job_id"])
            session_id = str(uuid.uuid4())
            store.complete(queued["job_id"], session_id)

            status = owner.get(queued["status_url"]).get_json()
            result = owner.get(status["result_url"])

        self.assertEqual(status["status"], "completed")
        self.assertEqual(result.status_code, 200)
        self.assertIn("Completed evidence dashboard", result.get_data(as_text=True))
        load_session.assert_called_once_with(session_id)

    def test_demo_job_completes_end_to_end_in_background(self):
        state_directory = self.temporary_directory.name
        with patch.dict(os.environ, {"DATA_PRISM_STATE_DIR": state_directory}):
            with web_app.app.test_client() as client:
                queued = client.post(
                    "/vibedash/jobs",
                    data={
                        "demo_dataset": "saas_growth",
                        "prompt": "Show evidence-backed SaaS performance.",
                    },
                )
                self.assertEqual(queued.status_code, 202)
                job = queued.get_json()

                status = None
                for _ in range(100):
                    status_response = client.get(job["status_url"])
                    self.assertEqual(status_response.status_code, 200)
                    status = status_response.get_json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.05)

                self.assertEqual(status["status"], "completed")
                result = client.get(status["result_url"])
                history = client.get("/vibedash/history")
                manifest = client.get(status["manifest_url"])

        self.assertEqual(result.status_code, 200)
        self.assertIn("SaaS Growth Evidence Dashboard", result.get_data(as_text=True))
        self.assertIn("Reproducibility record", result.get_data(as_text=True))
        self.assertEqual(history.status_code, 200)
        self.assertIn("Analysis history", history.get_data(as_text=True))
        self.assertIn("Dataset SHA-256", history.get_data(as_text=True))
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.get_json()["manifest_version"], 2)
        self.assertIn("attachment", manifest.headers["Content-Disposition"])

    @patch("vibedash.routes.analysis_job_dispatcher.submit", return_value=True)
    def test_history_and_manifest_are_isolated_by_browser_scope(self, _submit):
        with web_app.app.test_client() as owner:
            queued = owner.post(
                "/vibedash/jobs",
                data={"demo_dataset": "saas_growth", "prompt": "Analyze"},
            ).get_json()
            store = AnalysisJobStore(web_app.app.config["VIBEDASH_JOB_STORE_PATH"])
            store.claim(queued["job_id"])
            store.complete(
                queued["job_id"],
                str(uuid.uuid4()),
                {"manifest_version": 1},
            )
            owner_history = owner.get("/vibedash/history")

        manifest_url = queued["status_url"] + "/manifest"
        with web_app.app.test_client() as stranger:
            stranger_history = stranger.get("/vibedash/history")
            stranger_manifest = stranger.get(manifest_url)

        self.assertIn(queued["job_id"][:10], owner_history.get_data(as_text=True))
        self.assertNotIn(
            queued["job_id"][:10],
            stranger_history.get_data(as_text=True),
        )
        self.assertEqual(stranger_manifest.status_code, 404)


if __name__ == "__main__":
    unittest.main()
