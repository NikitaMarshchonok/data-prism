import math
import os
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from flask import Flask, render_template

import web_app
from src.demo_data import create_saas_growth_demo
from vibedash.audit_report import (
    MAX_FINDINGS,
    REPORT_CONTRACT,
    build_audit_report_context,
    clean_text,
)
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.spec import create_saas_demo_viz_spec


ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


class AuditReportContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = create_saas_growth_demo()
        spec = create_saas_demo_viz_spec()
        cls.temporal = generate_dashboard_data(frame, spec, prompt="Why does revenue differ?")
        cls.cross_sectional = generate_dashboard_data(
            frame.drop(columns=["Date"]), spec, prompt="Why does revenue differ?"
        )

    def session(self, dashboard, **extra):
        data = {
            "filename": "saas.csv",
            "prompt": "Why does revenue differ?",
            "dashboard_data": dashboard,
            "audit_manifest": {
                "analysis_contract": "vibedash-evidence-v3",
                "engine_version": "test",
                "dataset": {"content_sha256": "a" * 64, "analyzed_rows": 360, "truncated": False},
            },
        }
        data.update(extra)
        return data

    def test_report_is_deterministic_for_the_same_analysis(self):
        first = build_audit_report_context(self.session(self.temporal), generated_at=FIXED_TIME)
        second = build_audit_report_context(self.session(self.temporal), generated_at=FIXED_TIME)
        self.assertEqual(first, second)
        self.assertEqual(first["contract"], REPORT_CONTRACT)
        self.assertEqual(first["generated_at"], "2026-11-21 12:00 UTC")

    def test_readiness_lists_only_issues_with_fix_guidance(self):
        report = build_audit_report_context(self.session(self.temporal), generated_at=FIXED_TIME)
        readiness = report["readiness"]
        self.assertEqual(readiness["score"], self.temporal["readiness"]["score"])
        self.assertTrue(readiness["issues"])
        for issue in readiness["issues"]:
            self.assertIn(issue["status"], {"blocked", "warning"})
            self.assertTrue(issue["recommendation"])

    def test_findings_skip_coverage_and_put_warnings_first(self):
        report = build_audit_report_context(self.session(self.temporal), generated_at=FIXED_TIME)
        findings = report["findings"]
        self.assertLessEqual(len(findings), MAX_FINDINGS)
        self.assertNotIn("coverage", {item["category"] for item in findings})
        severities = [item["severity"] for item in findings]
        self.assertEqual(severities, sorted(severities, key=lambda s: {"warning": 1, "info": 2}.get(s, 3)))

    def test_withheld_inference_is_reported_without_tests(self):
        report = build_audit_report_context(self.session(self.temporal), generated_at=FIXED_TIME)
        self.assertTrue(report["statistics"]["withheld"])
        self.assertEqual(report["statistics"]["tests"], [])

    def test_valid_inference_exposes_corrected_p_values_and_intervals(self):
        report = build_audit_report_context(self.session(self.cross_sectional), generated_at=FIXED_TIME)
        statistics = report["statistics"]
        self.assertFalse(statistics["withheld"])
        self.assertIn("Benjamini-Hochberg", statistics["correction"])
        self.assertTrue(statistics["tests"])
        first = statistics["tests"][0]
        self.assertTrue(first["interval"].startswith("["))
        self.assertNotEqual(first["adjusted_p_value"], "n/a")

    def test_cover_labels_are_bounded_and_whitespace_collapsed(self):
        report = build_audit_report_context(
            self.session(self.temporal),
            prepared_for="  Acme\n\tCorp  ",
            prepared_by="x" * 500,
            generated_at=FIXED_TIME,
        )
        self.assertEqual(report["prepared_for"], "Acme Corp")
        self.assertLessEqual(len(report["prepared_by"]), 120)

    def test_missing_or_malformed_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            build_audit_report_context({"filename": "x.csv"})
        with self.assertRaises(ValueError):
            build_audit_report_context(None)

    def test_partial_dashboard_still_builds(self):
        report = build_audit_report_context(
            {"dashboard_data": {"insights": "not-a-list", "readiness": {"checks": [None]}}},
            generated_at=FIXED_TIME,
        )
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["readiness"]["issues"], [])
        self.assertEqual(report["dataset"]["name"], "Uploaded dataset")

    def test_blocked_readiness_is_reported_as_must_fix(self):
        dashboard = {
            "readiness": {
                "status": "blocked",
                "score": 20,
                "counts": {"passed": 1},
                "checks": [
                    {"status": "passed", "title": "Coverage", "recommendation": "None."},
                    {"status": "warning", "title": "Missing", "recommendation": "Review."},
                    {"status": "blocked", "title": "Empty target", "recommendation": "Upload data."},
                ],
            }
        }
        report = build_audit_report_context({"dashboard_data": dashboard}, generated_at=FIXED_TIME)
        readiness = report["readiness"]
        self.assertEqual(readiness["tone"], "bad")
        self.assertEqual([i["status"] for i in readiness["issues"]], ["blocked", "warning"])
        self.assertEqual(readiness["issues"][0]["status_label"], "Must fix")

    def test_non_finite_statistics_render_as_not_available(self):
        dashboard = {
            "statistical_validation": {
                "status": "ok",
                "alpha": math.nan,
                "tests": [
                    {
                        "title": "Broken test",
                        "adjusted_p_value": math.nan,
                        "effect_size": {"name": "r", "value": math.inf, "magnitude": "large"},
                        "confidence_interval": {"lower": -math.inf, "upper": 0.5},
                    }
                ],
            }
        }
        report = build_audit_report_context({"dashboard_data": dashboard}, generated_at=FIXED_TIME)
        test = report["statistics"]["tests"][0]
        self.assertEqual((test["adjusted_p_value"], test["effect"], test["interval"]), ("n/a", "n/a", "n/a"))
        self.assertIsNone(report["statistics"]["alpha"])

    def test_date_does_not_depend_on_server_locale(self):
        report = build_audit_report_context({"dashboard_data": {}}, generated_at=FIXED_TIME)
        self.assertEqual(report["generated_on"], "21 November 2026")

    def test_clean_text_truncates_with_ellipsis(self):
        self.assertEqual(clean_text("abcdef", maximum=4), "abc…")
        self.assertEqual(clean_text(None), "")

    def test_template_escapes_untrusted_labels(self):
        app = Flask(__name__, template_folder=str(ROOT / "templates"))
        report = build_audit_report_context(
            self.session(self.temporal, filename="<script>alert(1)</script>.csv"),
            prepared_for="<img src=x onerror=alert(1)>",
            generated_at=FIXED_TIME,
        )
        with app.test_request_context("/"):
            html = render_template("vibedash_audit_report.html", report=report, report_css="")
        self.assertNotIn("<script>alert(1)", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;img src=x", html)
        self.assertIn("Data Audit Report", html)


class AuditReportRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.previous = {
            key: web_app.app.config[key]
            for key in ("UPLOAD_FOLDER", "VIBEDASH_RETENTION_HOURS", "VIBEDASH_JOB_STORE_PATH")
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=self.temporary_directory.name,
            VIBEDASH_RETENTION_HOURS=24,
            VIBEDASH_JOB_STORE_PATH=str(Path(self.temporary_directory.name) / "analysis_jobs.sqlite3"),
        )

    def tearDown(self):
        web_app.app.config.update(self.previous)
        self.temporary_directory.cleanup()

    def run_demo(self, client, state_directory):
        response = client.post(
            "/vibedash/preview",
            data={"demo_dataset": "saas_growth", "prompt": "Show evidence-backed SaaS performance."},
        )
        self.assertEqual(response.status_code, 200)
        sessions = list((Path(state_directory) / "sessions" / "vibedash").glob("*.json"))
        self.assertEqual(len(sessions), 1)
        return sessions[0].stem

    def test_owner_downloads_pdf_and_html_preview(self):
        with TemporaryDirectory() as state_directory:
            with patch.dict(os.environ, {"DATA_PRISM_STATE_DIR": state_directory}):
                with web_app.app.test_client() as client:
                    session_id = self.run_demo(client, state_directory)
                    pdf = client.get(f"/vibedash/report/{session_id}.pdf?prepared_for=Acme")
                    preview = client.get(
                        f"/vibedash/report/{session_id}.pdf?format=html&prepared_for=Acme"
                    )

        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.headers["Content-Type"], "application/pdf")
        self.assertTrue(pdf.data.startswith(b"%PDF"))
        self.assertIn("attachment", pdf.headers["Content-Disposition"])
        self.assertEqual(pdf.headers["Cache-Control"], "private, no-store")
        self.assertEqual(preview.status_code, 200)
        page = preview.get_data(as_text=True)
        self.assertIn("Acme", page)
        self.assertIn("Executive summary", page)
        self.assertIn("saas_growth_demo.csv", page)

    def test_other_visitor_cannot_download_the_report(self):
        with TemporaryDirectory() as state_directory:
            with patch.dict(os.environ, {"DATA_PRISM_STATE_DIR": state_directory}):
                with web_app.app.test_client() as owner:
                    session_id = self.run_demo(owner, state_directory)
                with web_app.app.test_client() as stranger:
                    response = stranger.get(f"/vibedash/report/{session_id}.pdf")
        self.assertEqual(response.status_code, 404)

    def test_unknown_or_malformed_session_id_is_not_found(self):
        with web_app.app.test_client() as client:
            for session_id in ("not-a-session", "00000000-0000-4000-8000-000000000000"):
                response = client.get(f"/vibedash/report/{session_id}.pdf")
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.get_json(), {"error": "Analysis not found."})

    def test_html_preview_cannot_run_scripts_or_be_framed(self):
        with TemporaryDirectory() as state_directory:
            with patch.dict(os.environ, {"DATA_PRISM_STATE_DIR": state_directory}):
                with web_app.app.test_client() as client:
                    session_id = self.run_demo(client, state_directory)
                    preview = client.get(f"/vibedash/report/{session_id}.pdf?format=html")
        self.assertEqual(preview.headers["X-Frame-Options"], "DENY")
        self.assertIn("script-src 'none'", preview.headers["Content-Security-Policy"])
        self.assertIn("frame-ancestors 'none'", preview.headers["Content-Security-Policy"])

    def test_pdf_rendering_returns_retryable_busy_when_at_capacity(self):
        with TemporaryDirectory() as state_directory:
            with patch.dict(os.environ, {"DATA_PRISM_STATE_DIR": state_directory}):
                with web_app.app.test_client() as client:
                    session_id = self.run_demo(client, state_directory)
                    with patch("vibedash.routes.PDF_RENDER_SLOTS", threading.BoundedSemaphore(1)) as slots, \
                            patch("vibedash.routes.PDF_RENDER_WAIT_SECONDS", 0.01):
                        slots.acquire()
                        busy = client.get(f"/vibedash/report/{session_id}.pdf")
                        slots.release()
                        recovered = client.get(f"/vibedash/report/{session_id}.pdf")
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.headers["Retry-After"], "5")
        self.assertEqual(recovered.status_code, 200)
        self.assertTrue(recovered.data.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
