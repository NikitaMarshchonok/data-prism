"""Release acceptance journey for the supervised Data Prism pilot.

This test intentionally crosses route and storage boundaries.  Narrow unit and
integration tests still own edge cases; this journey protects the primary user
contract from registration through verified account deletion.
"""

from __future__ import annotations

import re
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app

from vibedash import routes
from vibedash.account_export import ACCOUNT_EXPORT_CONTRACT
from vibedash.accounts import RECOVERY_CODE_COUNT, account_scope_id
from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.decision_cases import DecisionCaseStore


class ReleaseAcceptanceJourneyTests(unittest.TestCase):
    """Exercise the critical pilot path against isolated real stores."""

    EMAIL = "release-acceptance@example.com"
    PASSWORD = "release acceptance password"
    NEW_PASSWORD = "replacement release password"

    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.account_path = self.root / "accounts" / "accounts.sqlite3"
        self.job_path = self.root / "jobs" / "analysis_jobs.sqlite3"
        self.previous_config = {
            key: web_app.app.config[key]
            for key in (
                "TESTING",
                "UPLOAD_FOLDER",
                "REPORT_FOLDER",
                "BASELINE_FOLDER",
                "VIBEDASH_ACCOUNT_STORE_PATH",
                "VIBEDASH_JOB_STORE_PATH",
                "VIBEDASH_RETENTION_HOURS",
                "VIBEDASH_JOB_TIMEOUT_SECONDS",
                "VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE",
                "VIBEDASH_MAX_ACTIVE_JOBS",
                "VIBEDASH_DECISION_RETENTION_DAYS",
                "VIBEDASH_MAX_DECISION_CASES_PER_SCOPE",
                "SESSION_COOKIE_SECURE",
            )
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(self.root / "uploads"),
            REPORT_FOLDER=str(self.root / "reports"),
            BASELINE_FOLDER=str(self.root / "baselines"),
            VIBEDASH_ACCOUNT_STORE_PATH=str(self.account_path),
            VIBEDASH_JOB_STORE_PATH=str(self.job_path),
            VIBEDASH_RETENTION_HOURS=24,
            VIBEDASH_JOB_TIMEOUT_SECONDS=600,
            VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE=2,
            VIBEDASH_MAX_ACTIVE_JOBS=25,
            VIBEDASH_DECISION_RETENTION_DAYS=90,
            VIBEDASH_MAX_DECISION_CASES_PER_SCOPE=50,
            SESSION_COOKIE_SECURE=False,
        )
        for directory in (
            self.root / "uploads",
            self.root / "reports",
            self.root / "baselines",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        old_store = web_app.app.extensions.pop("vibedash_account_store", None)
        if old_store is not None:
            old_store.close()

        self.environment = patch.dict(
            "os.environ",
            {"DATA_PRISM_STATE_DIR": str(self.root)},
        )
        self.environment.start()
        self.dispatcher = patch.object(
            routes.analysis_job_dispatcher,
            "submit",
            side_effect=self._run_job_inline,
        )
        self.dispatcher.start()

    def tearDown(self):
        self.dispatcher.stop()
        store = web_app.app.extensions.pop("vibedash_account_store", None)
        if store is not None:
            store.close()
        web_app.app.config.update(self.previous_config)
        self.environment.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def _csrf(client, path: str) -> str:
        response = client.get(path)
        if response.status_code != 200:
            raise AssertionError(
                f"Could not load CSRF form {path}: HTTP {response.status_code}"
            )
        match = re.search(
            r'name="csrf_token" value="([0-9a-f]{64})"',
            response.get_data(as_text=True),
        )
        if match is None:
            raise AssertionError(f"CSRF token missing from {path}")
        return match.group(1)

    @staticmethod
    def _recovery_codes(response) -> list[str]:
        return re.findall(
            r"\b[A-HJ-NP-Z2-9]{4}(?:-[A-HJ-NP-Z2-9]{4}){4}\b",
            response.get_data(as_text=True),
        )

    @staticmethod
    def _run_job_inline(application, job_id, processor) -> bool:
        """Use the production dispatcher lifecycle without a background race."""
        routes.analysis_job_dispatcher._run(application, job_id, processor)
        return True

    def _register(self, client):
        return client.post(
            "/vibedash/register",
            data={
                "csrf_token": self._csrf(client, "/vibedash/register"),
                "email": self.EMAIL,
                "password": self.PASSWORD,
            },
        )

    def _login(self, client, password):
        return client.post(
            "/vibedash/login",
            data={
                "csrf_token": self._csrf(client, "/vibedash/login"),
                "email": self.EMAIL,
                "password": password,
            },
        )

    def test_account_analysis_decision_recovery_and_deletion_journey(self):
        owner = web_app.app.test_client()

        registered = self._register(owner)
        self.assertEqual(registered.status_code, 302)
        self.assertEqual(registered.location, "/vibedash/")
        settings = owner.get("/vibedash/account")
        self.assertEqual(settings.status_code, 200)
        self.assertIn(self.EMAIL, settings.get_data(as_text=True))
        with owner.session_transaction() as browser:
            account_id = browser["vibedash_account_id"]
        scope_id = account_scope_id(account_id, web_app.app.secret_key)

        queued = owner.post(
            "/vibedash/jobs",
            data={
                "demo_dataset": "saas_growth",
                "prompt": "Review SaaS performance with evidence and limitations.",
            },
        )
        self.assertEqual(queued.status_code, 202)
        queued_payload = queued.get_json()
        job_id = queued_payload["job_id"]
        status = owner.get(queued_payload["status_url"])
        self.assertEqual(status.status_code, 200)
        status_payload = status.get_json()
        self.assertEqual(status_payload["status"], "completed")

        job_store = AnalysisJobStore(self.job_path)
        completed_job = job_store.get(job_id, scope_id)
        self.assertIsNotNone(completed_job)
        session_id = completed_job["session_id"]
        result = owner.get(status_payload["result_url"])
        result_html = result.get_data(as_text=True)
        self.assertEqual(result.status_code, 200)
        self.assertIn("SaaS Growth Evidence Dashboard", result_html)
        self.assertIn("Decision brief", result_html)

        manifest = owner.get(status_payload["manifest_url"])
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(
            manifest.get_json()["analysis_contract"],
            "vibedash-evidence-v2",
        )

        review_date = (date.today() + timedelta(days=30)).isoformat()
        decision = owner.post(
            f"/vibedash/jobs/{job_id}/decisions",
            data={
                "csrf_token": self._csrf(owner, status_payload["result_url"]),
                "priority": "1",
                "owner": "Pilot owner",
                "decision": "Test the highest-priority recommendation.",
                "success_metric": "Qualified activation rate",
                "target_outcome": "Improve the rate from 40% to 48%.",
                "review_date": review_date,
            },
        )
        self.assertEqual(decision.status_code, 303)
        case_id = decision.location.rstrip("/").split("/")[-1]

        outcome = owner.post(
            f"/vibedash/decisions/{case_id}/outcome",
            data={
                "csrf_token": self._csrf(owner, decision.location),
                "status": "validated",
                "actual_outcome": "The measured rate reached 48%.",
                "baseline_value": "40",
                "observed_value": "48",
                "outcome_unit": "percent",
                "observation_start": "2026-09-01",
                "observation_end": "2026-09-30",
            },
        )
        self.assertEqual(outcome.status_code, 303)
        detail_html = owner.get(outcome.location).get_data(as_text=True)
        self.assertIn("The measured rate reached 48%.", detail_html)
        self.assertIn("+20.0%", detail_html)
        closed_queue = owner.get("/vibedash/decisions?view=closed")
        self.assertEqual(closed_queue.status_code, 200)
        self.assertIn(
            "Test the highest-priority recommendation.",
            closed_queue.get_data(as_text=True),
        )
        self.assertIn(job_id[:10], owner.get("/vibedash/history").get_data(as_text=True))

        generated = owner.post(
            "/vibedash/account/recovery-codes",
            data={
                "csrf_token": self._csrf(owner, "/vibedash/account"),
                "current_password": self.PASSWORD,
            },
        )
        self.assertEqual(generated.status_code, 200)
        codes = self._recovery_codes(generated)
        self.assertEqual(len(codes), RECOVERY_CODE_COUNT)

        exported = owner.post(
            "/vibedash/account/export.json",
            data={"csrf_token": self._csrf(owner, "/vibedash/account")},
        )
        self.assertEqual(exported.status_code, 200)
        export_payload = exported.get_json()
        self.assertEqual(export_payload["contract"], ACCOUNT_EXPORT_CONTRACT)
        self.assertEqual(export_payload["jobs"][0]["id"], job_id)
        self.assertEqual(export_payload["decision_cases"][0]["id"], case_id)
        self.assertEqual(
            export_payload["decision_cases"][0]["outcome_measurement"][
                "delta_value"
            ],
            "8",
        )

        logged_out = owner.post(
            "/vibedash/logout",
            data={"csrf_token": self._csrf(owner, "/vibedash/account")},
        )
        self.assertEqual(logged_out.status_code, 302)
        self.assertIn("/vibedash/login", owner.get("/vibedash/account").location)
        self.assertEqual(self._login(owner, self.PASSWORD).status_code, 302)
        self.assertIn(job_id[:10], owner.get("/vibedash/history").get_data(as_text=True))

        recovery_browser = web_app.app.test_client()
        recovered = recovery_browser.post(
            "/vibedash/recover",
            data={
                "csrf_token": self._csrf(recovery_browser, "/vibedash/recover"),
                "email": self.EMAIL.upper(),
                "recovery_code": codes[0].lower(),
                "new_password": self.NEW_PASSWORD,
                "new_password_confirmation": self.NEW_PASSWORD,
            },
        )
        self.assertEqual(recovered.status_code, 302)
        self.assertEqual(recovery_browser.get("/vibedash/account").status_code, 200)
        self.assertEqual(owner.get("/vibedash/account").status_code, 302)
        self.assertIn(
            job_id[:10],
            recovery_browser.get("/vibedash/history").get_data(as_text=True),
        )

        old_password_browser = web_app.app.test_client()
        self.assertEqual(
            self._login(old_password_browser, self.PASSWORD).status_code,
            401,
        )
        new_password_browser = web_app.app.test_client()
        self.assertEqual(
            self._login(new_password_browser, self.NEW_PASSWORD).status_code,
            302,
        )

        deleted = recovery_browser.post(
            "/vibedash/account/delete",
            data={
                "csrf_token": self._csrf(recovery_browser, "/vibedash/account"),
                "current_password": self.NEW_PASSWORD,
                "delete_confirmation": "DELETE",
            },
        )
        self.assertEqual(deleted.status_code, 302)
        self.assertEqual(deleted.location, "/vibedash/")
        self.assertEqual(recovery_browser.get("/vibedash/account").status_code, 302)
        self.assertEqual(new_password_browser.get("/vibedash/account").status_code, 302)

        account_store = web_app.app.extensions["vibedash_account_store"]
        self.assertIsNone(account_store.get_account(account_id))
        self.assertIsNone(job_store.get(job_id))
        self.assertEqual(DecisionCaseStore(self.job_path).list_for_scope(scope_id), [])
        self.assertFalse(
            (self.root / "sessions" / "vibedash" / f"{session_id}.json").exists()
        )
        self.assertEqual(list((self.root / "uploads").glob("vibedash-*.csv")), [])


if __name__ == "__main__":
    unittest.main()
