import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app

from vibedash.decision_cases import DecisionCaseStore
from vibedash.decision_report import build_decision_report_context


class DecisionReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "jobs.sqlite3"
        self.previous_path = web_app.app.config["VIBEDASH_JOB_STORE_PATH"]
        web_app.app.config.update(
            TESTING=True,
            VIBEDASH_JOB_STORE_PATH=str(self.database_path),
        )
        self.scope_id = uuid.uuid4().hex
        self.store = DecisionCaseStore(self.database_path)

    def tearDown(self):
        web_app.app.config["VIBEDASH_JOB_STORE_PATH"] = self.previous_path
        self.temporary_directory.cleanup()

    def owner_client(self):
        client = web_app.app.test_client()
        with client.session_transaction() as browser_session:
            browser_session["vibedash_analysis_scope_id"] = self.scope_id
        return client

    def create_case(self, *, snapshot=None, decision="Test <script>alert(1)</script>."):
        job_id = uuid.uuid4().hex
        if snapshot is None:
            snapshot = {
                "contract": "decision-case-source-v1",
                "analysis_job_id": job_id,
                "analysis_contract": "evidence-dashboard-v2",
                "dataset_sha256": "b" * 64,
                "source_rows": ["must-not-appear"],
                "priority": {
                    "number": 1,
                    "title": "Investigate <activation> decline",
                    "finding": "Activation declined by 8 percentage points.",
                    "recommended_action": "Test a shorter onboarding path.",
                    "confidence": "high",
                    "evidence": ["Activation: 42% versus 50% baseline."],
                },
            }
        return self.store.create(
            self.scope_id,
            job_id,
            priority=1,
            owner="Growth lead",
            decision=decision,
            success_metric="Activation rate",
            target_outcome="Increase activation from 42% to 47%.",
            review_date="2026-10-15",
            evidence_snapshot=snapshot,
        )

    def test_context_is_bounded_and_excludes_unapproved_snapshot_fields(self):
        decision_case = self.create_case()

        report = build_decision_report_context(decision_case)

        self.assertEqual(report["contract"], "decision-case-report-v1")
        self.assertEqual(report["status_label"], "Tracking")
        self.assertEqual(report["evidence"]["dataset_sha256"], "b" * 64)
        self.assertNotIn("source_rows", report)
        self.assertNotIn("must-not-appear", str(report))

    def test_context_rejects_malformed_retained_contracts(self):
        decision_case = self.create_case()
        invalid_cases = (
            {**decision_case, "status": "unknown"},
            {**decision_case, "review_date": "tomorrow"},
            {**decision_case, "created_at": "unknown"},
            {**decision_case, "evidence_snapshot": []},
            {
                **decision_case,
                "evidence_snapshot": {
                    "contract": "unknown",
                    "priority": {},
                },
            },
        )
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_decision_report_context(invalid)

    def test_owner_downloads_standalone_safe_report(self):
        decision_case = self.create_case()

        response = self.owner_client().get(
            f"/vibedash/decisions/{decision_case['id']}/report.html"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/html")
        self.assertEqual(response.headers["Cache-Control"], "private, no-store")
        self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'none'", response.headers["Content-Security-Policy"])
        self.assertIn("script-src 'none'", response.headers["Content-Security-Policy"])
        self.assertIn(
            f'filename="data-prism-decision-{decision_case["id"][:12]}.html"',
            response.headers["Content-Disposition"],
        )
        html = response.get_data(as_text=True)
        self.assertIn("@media print", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("Investigate &lt;activation&gt; decline", html)
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("<form", html.lower())
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertNotIn("must-not-appear", html)
        self.assertNotIn("source_rows", html)

    def test_foreign_and_malformed_ids_are_non_leaking(self):
        decision_case = self.create_case()
        with web_app.app.test_client() as stranger:
            hidden = stranger.get(
                f"/vibedash/decisions/{decision_case['id']}/report.html"
            )
        malformed = self.owner_client().get(
            "/vibedash/decisions/not-a-case/report.html"
        )

        self.assertEqual(hidden.status_code, 404)
        self.assertEqual(malformed.status_code, 404)

    def test_invalid_report_contract_returns_conflict(self):
        decision_case = self.create_case(snapshot={"unexpected": True})

        response = self.owner_client().get(
            f"/vibedash/decisions/{decision_case['id']}/report.html"
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.get_json(),
            {"error": "Decision case report is unavailable."},
        )

    def test_stylesheet_failure_is_generic_and_style_end_is_escaped(self):
        decision_case = self.create_case()
        url = f"/vibedash/decisions/{decision_case['id']}/report.html"
        client = self.owner_client()
        with patch("vibedash.routes.Path.read_text", side_effect=OSError):
            failed = client.get(url)
        with patch(
            "vibedash.routes.Path.read_text",
            return_value="x </STYLE><script>bad</script> y",
        ):
            escaped = client.get(url)

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(
            failed.get_json(),
            {"error": "The decision case report could not be exported."},
        )
        escaped_html = escaped.get_data(as_text=True)
        self.assertEqual(escaped.status_code, 200)
        self.assertEqual(escaped_html.lower().count("</style>"), 1)
        self.assertIn("<\\/style><script>bad</script> y</style>", escaped_html)


if __name__ == "__main__":
    unittest.main()
