"""Do not confuse descriptive time-series signals with IID inference."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from flask import Flask, render_template

from src.column_semantics import profile_dataframe_columns
from src.semantic_contract import infer_semantic_contract, profiles_from_semantic_contract
from vibedash.ai_analyzer import DataScienceAI
from vibedash.audit_manifest import build_audit_manifest
from vibedash.account_export import _safe_manifest
from vibedash.decision_brief import build_decision_brief
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.insight_engine import EvidenceBasedInsightEngine
from vibedash.spec import Metric, VizSpec
from vibedash.statistical_engine import StatisticalValidationEngine


ROOT = Path(__file__).resolve().parents[1]


def _frame():
    x = np.linspace(10, 100, 120)
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=len(x)),
        "metric_x": x,
        "metric_y": 3 * x + np.sin(x),
        "group": ["A"] * 60 + ["B"] * 60,
    })


class TemporalInferenceGuardrailTests(unittest.TestCase):
    def test_dated_data_withholds_tests_before_invoking_iid_primitives(self):
        with patch("vibedash.statistical_engine.pearson_summary") as pearson, patch(
            "vibedash.statistical_engine.welch_independent_summary"
        ) as welch:
            report = StatisticalValidationEngine(_frame()).analyze()

        pearson.assert_not_called()
        welch.assert_not_called()
        self.assertEqual(report["status"], "inference_withheld")
        self.assertEqual(report["temporal_columns"], ["Date"])
        self.assertEqual(report["tests"], [])
        self.assertIsNone(report["significant_test_count"])
        self.assertIsNone(report["correction"])
        self.assertEqual(report["candidate_test_count"], 0)
        self.assertIn("significance is not assessed", report["summary"])
        json.dumps(report, allow_nan=False)

    def test_sorting_repeated_dates_and_missing_dates_do_not_bypass_policy(self):
        original = _frame()
        repeated = original.copy()
        repeated["Date"] = repeated["Date"].iloc[0]
        missing = original.copy()
        missing["Date"] = pd.NaT
        text = original.copy()
        text["Date"] = text["Date"].dt.strftime("%Y-%m-%d")
        for data in [original.sample(frac=1, random_state=7), repeated, missing, text]:
            with self.subTest(dtype=str(data["Date"].dtype)):
                report = StatisticalValidationEngine(data).analyze()
                self.assertEqual(report["inference_status"], "withheld")

    def test_confirmed_temporal_role_applies_without_selected_time_axis(self):
        data = _frame().rename(columns={"Date": "occasion"})
        contract = infer_semantic_contract(data, confirmed=True)
        contract["columns"][0]["is_time"] = False
        profiles = profiles_from_semantic_contract(data, contract)
        report = StatisticalValidationEngine(data, column_profiles=profiles).analyze()
        self.assertEqual(report["temporal_columns"], ["occasion"])
        self.assertEqual(report["inference_status"], "withheld")

    def test_temporal_role_not_column_name_controls_eligibility(self):
        data = _frame()
        profiles = profile_dataframe_columns(data)
        profiles["Date"]["role"] = "identifier"
        report = StatisticalValidationEngine(data, column_profiles=profiles).analyze()
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["tests"])
        self.assertTrue(report["tests"][0]["significant"])

    def test_no_time_field_keeps_existing_iid_oracle(self):
        data = pd.DataFrame({
            "metric_x": np.arange(1, 9, dtype=float),
            "metric_y": [2., 1., 4., 3., 6., 5., 8., 7.],
        })
        report = StatisticalValidationEngine(data).analyze()
        self.assertAlmostEqual(report["tests"][0]["raw_p_value"], 0.00200827550542948, places=14)
        self.assertTrue(any("independence" in note for note in report["methodology_notes"]))

    def test_empty_and_zero_requested_reports_keep_existing_behavior(self):
        self.assertEqual(StatisticalValidationEngine(_frame().iloc[:0]).analyze()["status"], "insufficient_data")
        self.assertEqual(StatisticalValidationEngine(_frame()).analyze(max_results=0)["tests"], [])

    def test_descriptive_relationships_and_changes_remain_exploratory(self):
        findings = EvidenceBasedInsightEngine(_frame()).generate(max_insights=10)
        selected = [finding for finding in findings if finding["category"] in {"relationship", "trend"}]
        self.assertEqual({finding["category"] for finding in selected}, {"relationship", "trend"})
        self.assertTrue(all(finding["confidence"] == "exploratory" for finding in selected))
        self.assertTrue(all("Descriptive observation only" in finding["evidence"][-1] for finding in selected))
        correlation = next(finding for finding in selected if finding["category"] == "relationship")
        self.assertAlmostEqual(correlation["metrics"]["pearson_r"], _frame()["metric_x"].corr(_frame()["metric_y"]), places=6)
        engine = StatisticalValidationEngine(_frame())
        self.assertEqual(engine.evidence_insights(), [])

    def test_withheld_report_cannot_promote_stale_significant_tests(self):
        report = StatisticalValidationEngine(_frame()).analyze()
        report["tests"] = [{"significant": True, "title": "Stale inference"}]
        engine = StatisticalValidationEngine(_frame())
        self.assertEqual(engine.evidence_insights(report), [])
        brief = build_decision_brief(
            {"statistical_validation": report, "insights": []},
            {"status": "ready", "status_label": "Ready", "checks": []},
        )
        self.assertEqual(brief["priorities"][0]["category"], "assumption-review")
        self.assertNotIn("validated-signal", [item["category"] for item in brief["priorities"]])

    def test_statistical_chat_explains_withholding_without_significance_claims(self):
        result = DataScienceAI(_frame()).analyze_question("run statistical tests")
        self.assertEqual(result["statistical_validation"]["inference_status"], "withheld")
        self.assertIn("independence", result["answer"])
        self.assertNotIn("FDR significant", result["answer"])
        self.assertFalse(any(item["category"] == "statistical-validation" for item in result["insights"]))
        self.assertTrue(all(
            item["confidence"] == "exploratory"
            for item in result["insights"] if item["category"] in {"relationship", "trend"}
        ))

    def test_dashboard_retains_kpis_and_prioritizes_sampling_design(self):
        data = _frame()
        spec = VizSpec(title="Temporal safety", metrics=[Metric(title="Sum", expr="sum(metric_x)")])
        dashboard = generate_dashboard_data(data, spec)
        self.assertEqual(dashboard["kpis"][0]["value"], "6,600.00")
        self.assertEqual(dashboard["df_shape"], data.shape)
        self.assertEqual(dashboard["statistical_validation"]["inference_status"], "withheld")
        self.assertEqual(dashboard["decision_brief"]["priorities"][0]["category"], "assumption-review")

    def test_html_templates_and_export_show_reason_and_escape_metadata(self):
        app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
        for endpoint in ["analysis_history", "decision_cases", "export", "index"]:
            app.add_url_rule("/" + endpoint, endpoint="vibedash." + endpoint, view_func=lambda: "")
        report = StatisticalValidationEngine(_frame()).analyze()
        report["methodology_notes"].append("<script>untrusted</script>")
        dashboard = {"df_shape": (120, 4), "statistical_validation": report}
        for template in ["vibedash_evidence.html", "vibedash_preview.html"]:
            for export_mode in [False, True]:
                with self.subTest(template=template, export=export_mode), app.test_request_context("/"):
                    html = render_template(template, dashboard_data=dashboard, viz_spec=VizSpec(title="Test"), filename="test.csv", prompt="Test", session_id="test", export_mode=export_mode)
                    self.assertIn("Statistical inference withheld", html)
                    self.assertIn("Sampling-design review required", html)
                    self.assertIn("significance is not assessed", html)
                    self.assertIn("&lt;script&gt;untrusted&lt;/script&gt;", html)
                    self.assertNotIn("<script>untrusted</script>", html)
                    self.assertNotIn("FDR significant", html)
                    self.assertNotIn("Adjusted p-value", html)

    def test_audit_and_account_export_do_not_turn_withholding_into_zero_discoveries(self):
        data = _frame()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            data.to_csv(source, index=False)
            manifest = build_audit_manifest(
                dataset_path=source, dataframe=data, source_row_count=len(data),
                truncated=False, source_name="data.csv", prompt="Analyze",
                demo_dataset="", run_id=None, viz_spec={},
                dashboard_data={"statistical_validation": StatisticalValidationEngine(data).analyze()},
                engine_version="test", retention_hours=24,
            )
        self.assertIsNone(manifest["evidence"]["significant_test_count"])
        self.assertEqual(manifest["evidence"]["statistical_inference_status"], "withheld")
        aggregate = _safe_manifest(manifest)
        self.assertEqual(aggregate["evidence"]["statistical_inference_status"], "withheld")
        self.assertNotIn("significant_test_count", aggregate["evidence"])
        json.dumps(aggregate, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
