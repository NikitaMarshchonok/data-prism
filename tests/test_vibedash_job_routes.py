import os
import time
import unittest
import uuid
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

        self.assertEqual(result.status_code, 200)
        self.assertIn("SaaS Growth Evidence Dashboard", result.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
