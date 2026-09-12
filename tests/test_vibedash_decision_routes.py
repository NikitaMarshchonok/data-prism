import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app

from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.decision_cases import DecisionCaseStore


class VibeDashDecisionRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.config_keys = (
            "UPLOAD_FOLDER",
            "VIBEDASH_JOB_STORE_PATH",
            "VIBEDASH_RETENTION_HOURS",
            "VIBEDASH_JOB_TIMEOUT_SECONDS",
            "VIBEDASH_DECISION_RETENTION_DAYS",
            "VIBEDASH_MAX_DECISION_CASES_PER_SCOPE",
        )
        self.previous_config = {
            key: web_app.app.config[key] for key in self.config_keys
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(root / "uploads"),
            VIBEDASH_JOB_STORE_PATH=str(root / "jobs.sqlite3"),
            VIBEDASH_RETENTION_HOURS=24,
            VIBEDASH_JOB_TIMEOUT_SECONDS=600,
            VIBEDASH_DECISION_RETENTION_DAYS=90,
            VIBEDASH_MAX_DECISION_CASES_PER_SCOPE=50,
        )
        Path(web_app.app.config["UPLOAD_FOLDER"]).mkdir()
        self.scope_id = uuid.uuid4().hex
        self.csrf_token = "a" * 64
        self.session_id = str(uuid.uuid4())
        self.job_store = AnalysisJobStore(
            web_app.app.config["VIBEDASH_JOB_STORE_PATH"]
        )
        job = self.job_store.create(self.scope_id, {"prompt": "Analyze activation"})
        self.job_store.claim(job["id"])
        self.job_store.complete(
            job["id"],
            self.session_id,
            {
                "analysis_contract": "evidence-dashboard-v2",
                "dataset": {"content_sha256": "b" * 64},
                "specification": {"title": "Activation review"},
            },
        )
        self.job_id = job["id"]
        self.session_data = {
            "filename": "saas.csv",
            "prompt": "Analyze activation",
            "viz_spec": {"title": "Activation review"},
            "dashboard_data": {
                "decision_brief": {
                    "priorities": [
                        {
                            "priority": 1,
                            "category": "growth",
                            "title": "Investigate activation decline",
                            "finding": "Activation declined by 8 percentage points.",
                            "action": "Test a shorter onboarding path.",
                            "confidence": "high",
                            "evidence": ["Activation: 42% versus 50% baseline."],
                        }
                    ]
                }
            },
        }

    def tearDown(self):
        web_app.app.config.update(self.previous_config)
        self.temporary_directory.cleanup()

    def owner_client(self):
        client = web_app.app.test_client()
        with client.session_transaction() as browser_session:
            browser_session["vibedash_analysis_scope_id"] = self.scope_id
            browser_session["vibedash_decision_csrf_token"] = self.csrf_token
        return client

    def valid_form(self):
        return {
            "csrf_token": self.csrf_token,
            "priority": "1",
            "owner": "Growth lead",
            "decision": "Test a shorter onboarding path.",
            "success_metric": "Activation rate",
            "target_outcome": "Increase activation from 42% to 47%.",
            "review_date": "2026-10-15",
        }

    @patch("vibedash.routes.load_session_data")
    def test_owner_can_create_and_open_evidence_linked_case(self, load_session):
        load_session.return_value = self.session_data
        client = self.owner_client()

        created = client.post(
            f"/vibedash/jobs/{self.job_id}/decisions",
            data=self.valid_form(),
        )

        self.assertEqual(created.status_code, 303)
        stored = DecisionCaseStore(
            web_app.app.config["VIBEDASH_JOB_STORE_PATH"]
        ).list_for_scope(self.scope_id)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["evidence_snapshot"]["dataset_sha256"], "b" * 64)
        self.assertNotIn("source rows", str(stored[0]).lower())

        detail = client.get(created.headers["Location"])
        self.assertEqual(detail.status_code, 200)
        self.assertIn("Investigate activation decline", detail.get_data(as_text=True))
        self.assertIn("Activation rate", detail.get_data(as_text=True))

        with web_app.app.test_client() as stranger:
            hidden = stranger.get(created.headers["Location"])
        self.assertEqual(hidden.status_code, 404)

    @patch("vibedash.routes.load_session_data")
    def test_creation_rejects_bad_token_and_unavailable_priority(self, load_session):
        load_session.return_value = self.session_data
        client = self.owner_client()
        bad_token = self.valid_form()
        bad_token["csrf_token"] = "wrong"

        token_response = client.post(
            f"/vibedash/jobs/{self.job_id}/decisions",
            data=bad_token,
        )
        bad_priority = self.valid_form()
        bad_priority["priority"] = "2"
        priority_response = client.post(
            f"/vibedash/jobs/{self.job_id}/decisions",
            data=bad_priority,
        )

        self.assertEqual(token_response.status_code, 400)
        self.assertEqual(priority_response.status_code, 400)
        self.assertEqual(
            DecisionCaseStore(
                web_app.app.config["VIBEDASH_JOB_STORE_PATH"]
            ).list_for_scope(self.scope_id),
            [],
        )

    @patch("vibedash.routes.load_session_data")
    def test_terminal_update_requires_observed_outcome(self, load_session):
        load_session.return_value = self.session_data
        client = self.owner_client()
        created = client.post(
            f"/vibedash/jobs/{self.job_id}/decisions",
            data=self.valid_form(),
        )
        case_id = created.headers["Location"].rstrip("/").split("/")[-1]

        missing = client.post(
            f"/vibedash/decisions/{case_id}/outcome",
            data={
                "csrf_token": self.csrf_token,
                "status": "validated",
                "actual_outcome": "",
            },
        )
        updated = client.post(
            f"/vibedash/decisions/{case_id}/outcome",
            data={
                "csrf_token": self.csrf_token,
                "status": "validated",
                "actual_outcome": "Activation reached 48% after four weeks.",
            },
        )

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(updated.status_code, 303)
        stored = DecisionCaseStore(
            web_app.app.config["VIBEDASH_JOB_STORE_PATH"]
        ).get(case_id, self.scope_id)
        self.assertEqual(stored["status"], "validated")
        self.assertEqual(stored["actual_outcome"], "Activation reached 48% after four weeks.")

    def test_empty_decision_workspace_is_rendered(self):
        response = self.owner_client().get("/vibedash/decisions")

        self.assertEqual(response.status_code, 200)
        self.assertIn("No decisions are being tracked", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
