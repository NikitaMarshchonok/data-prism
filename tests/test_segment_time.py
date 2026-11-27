import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template

from src.demo_data import create_saas_growth_demo
from vibedash.audit_report import build_audit_report_context
from vibedash.decision_brief import build_decision_brief
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.group_quality import _dated_segments, evaluate_segments_over_time
from vibedash.plain_language import explain_priority, explain_segment_gap, explain_statistics
from vibedash.segment_time_engine import (
    MIN_BATCHES,
    SegmentTimeEngine,
    batch_means,
    joint_batch_test,
    pair_batch_test,
)
from vibedash.spec import create_saas_demo_viz_spec

ROOT = Path(__file__).resolve().parents[1]
TEMPORAL = {"time_column": "Date"}


def _test(result, measure, segment="Plan"):
    return next(item for item in result["tests"] if item["measure"] == measure and item["segment_column"] == segment)


class BatchPrimitiveTests(unittest.TestCase):
    def test_batch_means_split_contiguously(self):
        values = np.arange(10.0)
        self.assertEqual(batch_means(values, 2).tolist(), [2.0, 7.0])

    def test_two_segment_joint_test_equals_the_batch_t_test(self):
        rng = np.random.default_rng(1)
        cells = rng.normal(0, 1, (40, 2)) + np.array([0.0, 0.4])
        statistic, p_joint, df1, df2 = joint_batch_test(cells, 8)
        _, p_pair, _ = pair_batch_test(cells[:, 1] - cells[:, 0], 8, 0.95)
        self.assertEqual((df1, df2), (1, 7))
        self.assertAlmostEqual(p_joint, p_pair, places=10)

    def test_singular_batches_return_none(self):
        cells = np.column_stack([np.arange(32.0), np.arange(32.0)])
        self.assertIsNone(joint_batch_test(cells, 8))


class SegmentTimeEngineTests(unittest.TestCase):
    def test_persistent_gap_is_found_and_named(self):
        frame = _dated_segments(np.random.default_rng(2), enterprise_shift=12.0)
        result = SegmentTimeEngine(frame, TEMPORAL).analyze()
        test = _test(result, "Revenue")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(test["significant"])
        self.assertEqual(test["highest_segment"], "Enterprise")
        self.assertEqual(test["grain"], "week")
        self.assertGreaterEqual(test["batches"], MIN_BATCHES)
        self.assertGreater(test["gap_ci"][0], 0)
        self.assertGreater(test["consistency"], 0.7)
        self.assertTrue(test["pairwise"])
        json.dumps(result)

    def test_shared_trend_and_season_do_not_create_a_gap(self):
        frame = _dated_segments(np.random.default_rng(3), phi=0.0)
        test = _test(SegmentTimeEngine(frame, TEMPORAL).analyze(), "Revenue")
        self.assertFalse(test["significant"])
        self.assertEqual(test["pairwise"], [])

    def test_yes_no_outcomes_are_compared_as_rates(self):
        frame = _dated_segments(np.random.default_rng(4), rate=True, enterprise_rate_shift=0.3)
        test = _test(SegmentTimeEngine(frame, TEMPORAL).analyze(), "Renewed")
        self.assertEqual(test["kind"], "rate")
        self.assertEqual(test["positive_level"], "Yes")
        self.assertEqual(test["unit"], "ratio")
        self.assertTrue(test["significant"])
        self.assertEqual(test["highest_segment"], "Enterprise")

    def test_segments_rarely_seen_together_are_skipped(self):
        rng = np.random.default_rng(5)
        dates = pd.date_range("2025-01-01", periods=120, freq="D")
        frame = pd.DataFrame({
            "Date": dates,
            "Region": np.where(np.arange(120) < 60, "North", "South"),
            "Revenue": rng.normal(100, 10, 120),
        })
        result = SegmentTimeEngine(frame, TEMPORAL).analyze()
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["skipped"][0]["reason"], "too_few_shared_periods")

    def test_without_a_time_column_nothing_runs(self):
        frame = _dated_segments(np.random.default_rng(6)).drop(columns=["Date"])
        self.assertEqual(SegmentTimeEngine(frame, {}).analyze()["status"], "not_applicable")

    def test_excluded_columns_are_respected(self):
        frame = _dated_segments(np.random.default_rng(7), enterprise_shift=12.0)
        result = SegmentTimeEngine(frame, TEMPORAL, analysis_context={"excluded_columns": ["Plan"]}).analyze()
        self.assertEqual(result["status"], "not_applicable")


class SegmentTimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = generate_dashboard_data(create_saas_growth_demo(), create_saas_demo_viz_spec())
        cls.app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
        for endpoint in ["analysis_history", "decision_cases", "export", "audit_report", "index"]:
            cls.app.add_url_rule("/" + endpoint, endpoint="vibedash." + endpoint, view_func=lambda: "")

    def test_demo_compares_plans_over_time_while_iid_tests_stay_withheld(self):
        segments = self.dashboard["segment_analysis"]
        self.assertEqual(self.dashboard["statistical_validation"]["inference_status"], "withheld")
        self.assertEqual(segments["status"], "ok")
        conversion = _test(segments, "ConversionRate")
        self.assertTrue(conversion["significant"])
        self.assertEqual(conversion["highest_segment"], "Enterprise")

    def test_brief_puts_the_gap_before_the_withheld_caveat_and_explains_it(self):
        categories = [item["category"] for item in self.dashboard["decision_brief"]["priorities"]]
        self.assertIn("segment-gap", categories)
        self.assertLess(categories.index("segment-gap"), categories.index("assumption-review"))
        priority = next(item for item in self.dashboard["decision_brief"]["priorities"] if item["category"] == "segment-gap")
        plain = explain_priority(priority, self.dashboard)
        self.assertIn("reliably higher", plain["headline"])
        caveat = next(item for item in self.dashboard["decision_brief"]["priorities"] if item["category"] == "assumption-review")
        self.assertIn("Segments were compared", explain_priority(caveat, self.dashboard)["headline"])

    def test_focus_on_another_measure_hides_the_gap(self):
        dashboard = dict(self.dashboard)
        dashboard["analysis_context"] = {"focus_columns": ["NPSScore"]}
        brief = build_decision_brief(dashboard, self.dashboard["readiness"])
        self.assertNotIn("segment-gap", [item["category"] for item in brief["priorities"]])

    def test_report_and_evidence_page_show_the_comparison(self):
        report = build_audit_report_context({"dashboard_data": self.dashboard})
        self.assertTrue(report["segments"]["shown"])
        self.assertIn("Segments were compared", report["statistics"]["plain"]["headline"])
        with self.app.test_request_context("/"):
            html = render_template("vibedash_audit_report.html", report=report, report_css="")
        self.assertIn("Segments compared over time", html)
        self.assertIn("reliably higher for “Enterprise”", html.replace("&#34;", '"'))

    def test_manifest_counts_segment_tests(self):
        from tempfile import TemporaryDirectory

        from vibedash.audit_manifest import build_audit_manifest

        frame = create_saas_growth_demo()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            frame.to_csv(source, index=False)
            manifest = build_audit_manifest(
                dataset_path=source, dataframe=frame, source_row_count=len(frame), truncated=False,
                source_name="data.csv", prompt="", demo_dataset="", run_id=None, viz_spec={},
                dashboard_data=self.dashboard, engine_version="test", retention_hours=24,
            )
        summary = manifest["evidence"]["segment_analysis"]
        self.assertEqual(summary["status"], "ok")
        self.assertGreaterEqual(summary["significant_test_count"], 1)


class SegmentTimePlainLanguageTests(unittest.TestCase):
    TEST = {
        "measure": "Renewed", "segment_column": "Plan", "kind": "rate", "positive_level": "Yes", "unit": "ratio",
        "grain": "week", "periods_compared": 43, "highest_segment": "Enterprise", "lowest_segment": "Starter",
        "highest_average": 0.9, "lowest_average": 0.7, "consistency": 0.81, "magnitude": "large", "significant": True,
    }

    def test_rate_gap_sentence(self):
        plain = explain_segment_gap(self.TEST)
        self.assertEqual(
            plain["headline"],
            "The share of “Renewed” = “Yes” is reliably higher for “Enterprise” (90.0% on average) than for “Starter” (70.0%).",
        )
        self.assertIn("43 weeks", plain["meaning"])
        self.assertIn("81%", plain["meaning"])

    def test_no_gap_sentence_does_not_claim_one(self):
        plain = explain_segment_gap(dict(self.TEST, significant=False))
        self.assertIn("does not differ reliably", plain["headline"])

    def test_withheld_text_without_segments_is_unchanged(self):
        validation = {"status": "inference_withheld", "reason": "temporal_design_unverified"}
        self.assertNotIn("Segments", explain_statistics(validation, temporal_checked=True)["headline"])


class SegmentTimeQualityGateTests(unittest.TestCase):
    def test_false_alarms_and_detection_meet_thresholds(self):
        report = evaluate_segments_over_time(seeds=150)
        failed = [item for item in report["scenarios"] if not item["passed"]]
        self.assertEqual(failed, [])
        persistent = next(item for item in report["scenarios"] if item["name"] == "dated_null_persistent")
        self.assertGreater(persistent["naive_rate"], 3 * persistent["rate"])


if __name__ == "__main__":
    unittest.main()
