import unittest
from pathlib import Path

from flask import Flask, render_template


ROOT = Path(__file__).resolve().parents[1]


def _ui_app():
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )
    endpoint_paths = {
        "vibedash.index": "/vibedash/",
        "vibedash.preview": "/vibedash/preview",
        "vibedash.dataset_readiness": "/vibedash/readiness",
        "vibedash.create_analysis_job": "/vibedash/jobs",
        "vibedash.create_period_comparison_job": "/vibedash/comparisons/jobs",
        "vibedash.analysis_history": "/vibedash/history",
        "vibedash.decision_cases": "/vibedash/decisions",
        "vibedash.forget_pilot_metrics": "/vibedash/pilot/forget",
        "upload_file": "/upload",
    }
    for endpoint, path in endpoint_paths.items():
        app.add_url_rule(path, endpoint=endpoint, view_func=lambda: "")
    return app


class VibeDashComparisonUITests(unittest.TestCase):
    def test_landing_exposes_separate_comparison_contract(self):
        app = _ui_app()
        with app.test_request_context("/vibedash/"):
            html = render_template(
                "vibedash_landing.html",
                pilot_csrf_token="token",
                demo_prompt="Demo",
                preset_prompts={"saas_review": "Review", "sales": "Sales", "finance": "Finance", "real_estate": "Property"},
                retention_hours=24,
                ollama_available=False,
            )
        self.assertIn('id="period-comparison-form"', html)
        self.assertIn('name="baseline_file"', html)
        self.assertIn('name="current_file"', html)
        self.assertIn('name="baseline_label"', html)
        self.assertIn('name="current_label"', html)
        self.assertIn('action="/vibedash/comparisons/jobs"', html)
        self.assertIn('const comparisonJobsUrl = "/vibedash/comparisons/jobs"', html)
        self.assertIn("same row unit and metric definitions", html)
        self.assertIn("independent observational comparison", html)
        self.assertIn("comparisonBusy", html)
        self.assertNotIn("innerHTML", html)

    def test_result_escapes_supplied_metadata_and_keeps_aggregate_contract(self):
        app = _ui_app()
        report = {
            "status": "stable",
            "summary": "No eligible numeric metrics were available.",
            "input": {
                "baseline": {"rows": 2, "columns": 2, "label": "ignored"},
                "current": {"rows": 2, "columns": 2},
            },
            "schema_changes": {"new_columns": [], "missing_columns": [], "type_changes": []},
            "numeric_metrics": {"metrics": []},
            "distribution_drift": {"signals": []},
            "limitations": ["This is observational and non-causal."],
        }
        malicious_label = "<script>alert('label')</script>"
        malicious_filename = "<img src=x onerror=alert('file')>.csv"
        with app.test_request_context("/vibedash/jobs/id/result"):
            html = render_template(
                "vibedash_comparison.html",
                report=report,
                baseline_filename=malicious_filename,
                current_filename="current.csv",
                baseline_label=malicious_label,
                current_label="Current",
                manifest_url="/vibedash/jobs/id/manifest",
            )
        self.assertIn("&lt;script&gt;alert(&#39;label&#39;)&lt;/script&gt;", html)
        self.assertIn("&lt;img src=x onerror=alert(&#39;file&#39;)&gt;.csv", html)
        self.assertNotIn(malicious_label, html)
        self.assertNotIn(malicious_filename, html)
        self.assertIn("Structure changes", html)
        self.assertIn("current_minus_baseline", html)
        self.assertIn("No eligible numeric metrics", html)
        self.assertIn("Limitations", html)
        self.assertIn("non-causal", html)
        self.assertIn("Open audit manifest", html)
        self.assertNotIn("raw_rows", html)
        self.assertNotIn("category_values", html)

    def test_result_renders_nullable_insufficient_metrics_as_dashes(self):
        app = _ui_app()
        report = {
            "status": "review_required",
            "summary": "One metric has insufficient evidence.",
            "input": {"baseline": {"rows": 2, "columns": 1}, "current": {"rows": 2, "columns": 1}},
            "schema_changes": {"new_columns": [], "missing_columns": [], "type_changes": []},
            "numeric_metrics": {
                "metrics": [{
                    "column": "value",
                    "baseline": {"mean": None, "median": None},
                    "current": {"mean": None, "median": None},
                    "absolute_change": None,
                    "percent_change": None,
                    "confidence_interval": None,
                    "adjusted_p_value": None,
                    "hedges_g": None,
                    "magnitude": None,
                    "status": "insufficient_evidence",
                }],
            },
            "distribution_drift": {"signals": [{"feature": "value", "score": None, "missing_rate_delta": None}]},
            "limitations": [],
        }
        with app.test_request_context("/vibedash/jobs/id/result"):
            html = render_template("vibedash_comparison.html", report=report, manifest_url=None)
        self.assertIn("insufficient_evidence", html)
        self.assertNotIn(">None<", html)
        self.assertNotIn("<br><span>None</span>", html)
        self.assertNotIn("<dd>None</dd>", html)

    def test_comparison_assets_are_separate_and_accessible(self):
        landing_css = (ROOT / "static" / "vibedash_landing.css").read_text()
        result_css = (ROOT / "static" / "vibedash_comparison.css").read_text()
        result_template = (ROOT / "templates" / "vibedash_comparison.html").read_text()
        self.assertIn(".period-comparison", landing_css)
        self.assertIn(".period-comparison-form", landing_css)
        self.assertIn("role=\"status\"", result_template)
        self.assertIn("aria-labelledby=\"metrics-heading\"", result_template)
        self.assertIn("@media", result_css)


if __name__ == "__main__":
    unittest.main()
