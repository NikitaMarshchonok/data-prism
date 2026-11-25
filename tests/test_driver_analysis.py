import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from flask import Flask, render_template

from vibedash.audit_manifest import build_audit_manifest
from vibedash.audit_report import build_audit_report_context
from vibedash.decision_brief import build_decision_brief
from vibedash.driver_analysis import (
    MAX_SEGMENTS_PER_DIMENSION,
    DriverAnalysisEngine,
    comparison_windows,
    decompose_mean,
    decompose_sum,
)
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.plain_language import explain_driver, explain_drivers, explain_priority
from vibedash.spec import VizSpec
from vibedash.temporal_patterns import level_shift_test
from vibedash.temporal_quality import _DRIVER_TEMPORAL, _driver_frame, evaluate_driver_analysis


ROOT = Path(__file__).resolve().parents[1]
READY = {"status": "ready", "status_label": "Ready", "checks": []}


def analyze(frame, temporal=_DRIVER_TEMPORAL, **kwargs):
    return DriverAnalysisEngine(frame, temporal, **kwargs).analyze()


def measure(result, column):
    return next(item for item in result["measures"] if item["column"] == column)


class WindowTests(unittest.TestCase):
    def test_weekly_windows_are_four_complete_weeks(self):
        pairs = comparison_windows("week", "2024-01-01", "2024-12-23")
        current_start, current_end, base_start, base_end = pairs[0]
        self.assertEqual((current_start, current_end), (pd.Timestamp("2024-12-02"), pd.Timestamp("2024-12-29")))
        self.assertEqual((base_start, base_end), (pd.Timestamp("2024-11-04"), pd.Timestamp("2024-12-01")))
        self.assertEqual(len(pairs), 12)
        self.assertTrue(all(pair[2] >= pd.Timestamp("2024-01-01") for pair in pairs))

    def test_daily_and_monthly_windows(self):
        daily = comparison_windows("day", "2024-01-01", "2024-03-31")[0]
        self.assertEqual(daily, (pd.Timestamp("2024-03-25"), pd.Timestamp("2024-03-31"), pd.Timestamp("2024-03-18"), pd.Timestamp("2024-03-24")))
        monthly = comparison_windows("month", "2023-01-01", "2024-12-01")
        self.assertEqual(monthly[0], (pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-01"), pd.Timestamp("2023-12-31")))
        self.assertEqual(len(monthly), 12)
        self.assertEqual(comparison_windows("month", "2024-01-01", "2024-12-01"), [])
        with self.assertRaises(ValueError):
            comparison_windows("quarter", "2024-01-01", "2024-12-01")


class DecompositionTests(unittest.TestCase):
    def test_sum_contributions_add_up_exactly(self):
        result = decompose_sum({"A": 100.0, "B": 50.0, "C": 10.0}, {"A": 60.0, "B": 55.0, "D": 5.0})
        self.assertAlmostEqual(sum(row["contribution"] for row in result["segments"]), result["change"])
        self.assertAlmostEqual(result["change"], -40.0)
        by_segment = {row["segment"]: row for row in result["segments"]}
        self.assertAlmostEqual(by_segment["A"]["relative_change"], -0.4)
        self.assertIsNone(by_segment["D"]["relative_change"])

    def test_mean_mix_and_rate_add_up_exactly(self):
        before = {"Pro": (70, 0.06), "Starter": (30, 0.02)}
        after = {"Pro": (50, 0.06), "Starter": (50, 0.02), "Trial": (10, 0.01)}
        result = decompose_mean(before, after)
        self.assertAlmostEqual(result["mix_effect"] + result["rate_effect"], result["change"], places=12)
        self.assertAlmostEqual(sum(row["contribution"] for row in result["segments"]), result["change"], places=12)
        expected_after = (50 * 0.06 + 50 * 0.02 + 10 * 0.01) / 110
        self.assertAlmostEqual(result["after"], expected_after)
        # Same segment rates, so the whole change is mix.
        self.assertAlmostEqual(result["rate_effect"], 0.0, places=12)
        starter = next(row for row in result["segments"] if row["segment"] == "Starter")
        # Gaining share while sitting below average lowers the average.
        self.assertLess(starter["mix"], 0)

    def test_simpsons_paradox_is_visible(self):
        # Every segment improves, but the average falls because of mix.
        before = {"Pro": (90, 0.060), "Starter": (10, 0.020)}
        after = {"Pro": (40, 0.065), "Starter": (60, 0.022)}
        result = decompose_mean(before, after)
        self.assertLess(result["change"], 0)
        self.assertGreater(result["rate_effect"], 0)
        self.assertLess(result["mix_effect"], result["change"])
        with self.assertRaises(ValueError):
            decompose_mean({}, after)


class EngineTests(unittest.TestCase):
    def test_disproportionate_segment_is_named(self):
        result = analyze(_driver_frame(np.random.default_rng(1), europe_drop=0.5))
        json.dumps(result, allow_nan=False)
        revenue = measure(result, "Revenue")
        self.assertEqual(revenue["kind"], "sum")
        self.assertTrue(revenue["material"])
        self.assertLess(revenue["relative_change"], -0.1)
        top = revenue["dimensions"][0]
        self.assertEqual(top["dimension"], "Region")
        self.assertEqual(top["driver"]["segment"], "Europe")
        self.assertGreater(top["driver"]["share_of_change"], 0.8)
        self.assertAlmostEqual(top["driver"]["base_share"], 0.3, delta=0.05)
        # Contributions of all segments add up to the total change.
        frame = _driver_frame(np.random.default_rng(1), europe_drop=0.5)
        late = frame[(frame["Date"] >= "2024-12-02")]
        prior = frame[(frame["Date"] >= "2024-11-04") & (frame["Date"] < "2024-12-02")]
        self.assertAlmostEqual(revenue["change"], late["Revenue"].sum() - prior["Revenue"].sum(), places=6)

    def test_mix_shift_is_recognized(self):
        result = analyze(_driver_frame(np.random.default_rng(2), starter_surge=True))
        conversion = measure(result, "Conversion")
        self.assertEqual(conversion["kind"], "average")
        top = conversion["dimensions"][0]
        self.assertEqual(top["dimension"], "Plan")
        self.assertTrue(top["mix_story"])
        self.assertEqual(top["mix_driver"]["segment"], "Starter")
        self.assertAlmostEqual(top["mix_effect"] + top["rate_effect"], conversion["change"], places=12)

    def test_flat_total_reports_offsetting_moves_without_a_driver(self):
        frame = _driver_frame(np.random.default_rng(3))
        late = frame["Date"] >= "2024-12-02"
        frame.loc[late & (frame["Region"] == "Europe"), "Revenue"] *= 0.5
        frame.loc[late & (frame["Region"] == "US"), "Revenue"] *= 1.3
        revenue = measure(analyze(frame), "Revenue")
        self.assertLess(abs(revenue["relative_change"]), 0.05)
        region = next(item for item in revenue["dimensions"] if item["dimension"] == "Region")
        self.assertIsNone(region["driver"])
        self.assertEqual({row["segment"] for row in region["offsetting"]}, {"Europe", "US"})
        self.assertTrue(all(row["share_of_change"] is None for row in region["segments"]))

    def test_signed_measures_have_no_percentage(self):
        frame = _driver_frame(np.random.default_rng(4))
        frame["Profit"] = frame["Revenue"] - 450
        temporal = dict(_DRIVER_TEMPORAL, series=[dict(_DRIVER_TEMPORAL["series"][0], column="Profit", unit="currency")])
        profit = measure(analyze(frame, temporal), "Profit")
        self.assertIsNone(profit["relative_change"])

    def test_statuses_and_context(self):
        frame = _driver_frame(np.random.default_rng(5))
        self.assertEqual(analyze(frame, {"status": "not_applicable"})["status"], "not_applicable")
        short = dict(_DRIVER_TEMPORAL, series=[dict(item, first_period="2024-11-25") for item in _DRIVER_TEMPORAL["series"]])
        self.assertEqual(analyze(frame, short)["status"], "insufficient_data")
        excluded = analyze(frame, analysis_context={"excluded_columns": ["Region", "Revenue"]})
        self.assertEqual([item["column"] for item in excluded["measures"]], ["Conversion"])
        self.assertNotIn("Region", excluded["dimensions"])
        focused = analyze(frame, analysis_context={"focus_columns": ["Conversion"]})
        self.assertEqual(focused["measures"][0]["column"], "Conversion")
        with self.assertRaises(TypeError):
            DriverAnalysisEngine([1, 2], _DRIVER_TEMPORAL)

    def test_many_segments_are_bucketed(self):
        frame = _driver_frame(np.random.default_rng(6))
        frame["Store"] = [f"S{index % 30}" for index in range(len(frame))]
        store = next(item for item in measure(analyze(frame), "Revenue")["dimensions"] if item["dimension"] == "Store")
        self.assertEqual(store["segment_count"], MAX_SEGMENTS_PER_DIMENSION + 1)
        frame["Store"] = [f"S{index % 60}" for index in range(len(frame))]
        dimensions = [item["dimension"] for item in measure(analyze(frame), "Revenue")["dimensions"]]
        self.assertNotIn("Store", dimensions)


class LevelShiftEdgeTests(unittest.TestCase):
    def test_change_in_the_last_periods_is_flagged_as_undatable(self):
        starts = list(pd.date_range("2024-01-01", periods=52, freq="W-MON"))
        values = 100 - 30 * (np.arange(52) >= 49) + np.random.default_rng(7).normal(0, 3, 52)
        self.assertTrue(level_shift_test(values, starts)["at_edge"])
        values = 100 - 30 * (np.arange(52) >= 26) + np.random.default_rng(7).normal(0, 3, 52)
        self.assertFalse(level_shift_test(values, starts)["at_edge"])


class DriverQualityGateTests(unittest.TestCase):
    def test_false_alarms_detection_and_attribution(self):
        report = evaluate_driver_analysis(seeds=100)
        for scenario in report["scenarios"]:
            with self.subTest(scenario=scenario["name"]):
                self.assertTrue(scenario["passed"], scenario)
        with self.assertRaises(ValueError):
            evaluate_driver_analysis(seeds=0)


class PlainLanguageTests(unittest.TestCase):
    def test_driver_mix_and_flat_wording(self):
        drop = measure(analyze(_driver_frame(np.random.default_rng(1), europe_drop=0.5)), "Revenue")
        plain = explain_driver(drop)
        self.assertRegex(plain["headline"], r"^“Revenue” fell by \d+%, from [\d,]+ to [\d,]+ \(last 4 weeks vs the 4 before\)\.$")
        self.assertIn("comes from “Europe” (“Region”)", plain["meaning"])
        self.assertIn("larger than the usual swing", plain["meaning"])

        mix = measure(analyze(_driver_frame(np.random.default_rng(2), starter_surge=True)), "Conversion")
        plain = explain_driver(mix)
        self.assertRegex(plain["headline"], r"^“Conversion” fell from an average of [\d.]+% to [\d.]+%")
        self.assertIn("mix effect", plain["meaning"])
        self.assertIn("“Plan” is “Starter”", plain["meaning"])

        flat = dict(drop, relative_change=0.01, change=10.0, dimensions=[])
        self.assertIn("roughly flat", explain_driver(flat)["headline"])
        self.assertIsNone(explain_driver(dict(drop, grain="quarter")))
        self.assertIsNone(explain_drivers({"status": "insufficient_data"}))


class ProductIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = _driver_frame(np.random.default_rng(1), europe_drop=0.5)
        cls.dashboard = generate_dashboard_data(frame, VizSpec(title="Drivers"))

    def test_dashboard_brief_and_plain_priority(self):
        drivers = self.dashboard["driver_analysis"]
        self.assertEqual(drivers["status"], "ok")
        json.dumps(drivers, allow_nan=False)
        priority = next(item for item in self.dashboard["decision_brief"]["priorities"] if item["category"] == "change-driver")
        self.assertEqual(priority["columns"], ["Revenue"])
        self.assertIn("Region = Europe", priority["finding"])
        self.assertIn("“Europe”", explain_priority(priority, self.dashboard)["meaning"])

    def test_brief_without_unusual_change_has_no_driver_priority(self):
        quiet = dict(self.dashboard["driver_analysis"])
        quiet["measures"] = [dict(item, unusual=dict(item["unusual"], significant=False)) for item in quiet["measures"]]
        brief = build_decision_brief({"driver_analysis": quiet}, READY)
        self.assertNotIn("change-driver", [item["category"] for item in brief["priorities"]])

    def test_report_and_page_sections(self):
        report = build_audit_report_context({"dashboard_data": self.dashboard})
        self.assertTrue(report["drivers"]["shown"])
        self.assertTrue(report["drivers"]["measures"][0]["material"])
        app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
        for endpoint in ["analysis_history", "decision_cases", "export", "audit_report", "index"]:
            app.add_url_rule("/" + endpoint, endpoint="vibedash." + endpoint, view_func=lambda: "")
        with app.test_request_context("/"):
            html = render_template("vibedash_audit_report.html", report=report, report_css="")
            page = render_template(
                "vibedash_evidence.html", dashboard_data=self.dashboard, viz_spec=VizSpec(title="Drivers"),
                filename="t.csv", prompt="", session_id="s", export_mode=False,
            )
        headings = re.findall(r"<h2>(\d+)\. ([^<]+)</h2>", html)
        self.assertIn(("5", "What changed recently, and where"), headings)
        self.assertIn(("6", "Statistical validation"), headings)
        self.assertIn("What changed recently", page)
        self.assertIn("driver: Europe", page)

    def test_manifest_records_driver_status(self):
        frame = _driver_frame(np.random.default_rng(1), europe_drop=0.5)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            frame.to_csv(source, index=False)
            manifest = build_audit_manifest(
                dataset_path=source, dataframe=frame, source_row_count=len(frame), truncated=False,
                source_name="data.csv", prompt="", demo_dataset="", run_id=None, viz_spec={},
                dashboard_data=self.dashboard, engine_version="test", retention_hours=24,
            )
        drivers = manifest["evidence"]["driver_analysis"]
        self.assertEqual(drivers["contract"], "driver-analysis-v1")
        self.assertGreaterEqual(drivers["material_change_count"], 1)

    def test_empty_table(self):
        empty = generate_dashboard_data(pd.DataFrame(), VizSpec(title="Empty"))
        self.assertEqual(empty["driver_analysis"]["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
