import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from flask import Flask, render_template

from src.column_semantics import profile_dataframe_columns
from src.demo_data import create_saas_growth_demo
from vibedash.audit_manifest import build_audit_manifest
from vibedash.audit_report import build_audit_report_context
from vibedash.decision_brief import build_decision_brief
from vibedash.generator_bridge import _without_superseded_trends, generate_dashboard_data
from vibedash.plain_language import (
    explain_priority,
    explain_statistics,
    explain_temporal,
    explain_temporal_series,
)
from vibedash.spec import VizSpec, create_saas_demo_viz_spec
from vibedash.temporal_engine import TemporalAnalysisEngine


ROOT = Path(__file__).resolve().parents[1]
READY = {"status": "ready", "status_label": "Ready", "checks": []}


def dated_frame(days=364, seed=11):
    rng = np.random.default_rng(seed)
    day = np.arange(days)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=days, freq="D"),
            "Revenue": 1000 + 2.0 * day + rng.normal(0, 40, days),
            "Visits": 500 + rng.normal(0, 30, days),
            "Profit": -200 + 1.5 * day + rng.normal(0, 20, days),
        }
    )


def series_item(**overrides):
    item = {
        "column": "Revenue",
        "grain": "week",
        "period_count": 50,
        "first_period": "2024-01-08",
        "last_period": "2024-12-16",
        "change_over_period": {"relative_change": 0.25, "relative_change_ci": [0.18, 0.33]},
        "trend": {"direction": "increasing", "significant": True, "adjusted_p_value": 0.001},
    }
    item.update(overrides)
    return item


class EngineIntegrationTests(unittest.TestCase):
    def test_units_follow_contract_then_inference(self):
        frame = dated_frame()
        result = TemporalAnalysisEngine(frame).analyze()
        by_column = {item["column"]: item for item in result["series"]}
        self.assertEqual(by_column["Revenue"]["unit"], "currency")
        self.assertEqual(by_column["Revenue"]["aggregation"], "sum")
        self.assertEqual(by_column["Visits"]["aggregation"], "mean")

        profiles = profile_dataframe_columns(frame)
        profiles["Visits"] = {**profiles["Visits"], "unit": "count"}
        confirmed = TemporalAnalysisEngine(frame, column_profiles=profiles).analyze()
        self.assertEqual({item["column"]: item["aggregation"] for item in confirmed["series"]}["Visits"], "sum")

        explicit = TemporalAnalysisEngine(frame, units={"Revenue": "ratio"}).analyze()
        self.assertEqual({item["column"]: item["aggregation"] for item in explicit["series"]}["Revenue"], "mean")

    def test_confirmed_time_axis_wins(self):
        frame = dated_frame()
        frame.insert(0, "Signup", pd.date_range("2020-01-01", periods=len(frame), freq="D"))
        profiles = profile_dataframe_columns(frame)
        profiles["Date"] = {**profiles["Date"], "is_time": True}
        result = TemporalAnalysisEngine(frame, column_profiles=profiles).analyze()
        self.assertEqual(result["time_column"], "Date")

    def test_change_over_period_is_relative_only_for_positive_levels(self):
        result = TemporalAnalysisEngine(dated_frame()).analyze()
        by_column = {item["column"]: item for item in result["series"]}
        revenue = by_column["Revenue"]["change_over_period"]
        # Fitted level grows from ~1000 to ~1700 per day, i.e. about +70%.
        self.assertAlmostEqual(revenue["relative_change"], 0.70, delta=0.08)
        low, high = revenue["relative_change_ci"]
        self.assertLess(low, revenue["relative_change"])
        self.assertGreater(high, revenue["relative_change"])
        profit = by_column["Profit"]["change_over_period"]
        self.assertIsNone(profit["relative_change"])
        self.assertIsNone(profit["relative_change_ci"])
        self.assertGreater(profit["absolute_change"], 0)


class DashboardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = generate_dashboard_data(dated_frame(), VizSpec(title="Trends"))

    def test_dashboard_carries_serializable_temporal_analysis(self):
        temporal = self.dashboard["temporal_analysis"]
        self.assertEqual(temporal["status"], "ok")
        json.dumps(temporal, allow_nan=False)
        empty = generate_dashboard_data(pd.DataFrame(), VizSpec(title="Empty"))
        self.assertEqual(empty["temporal_analysis"]["status"], "not_applicable")

    def test_descriptive_change_is_replaced_by_the_validated_trend(self):
        ids = [item["insight_id"] for item in self.dashboard["insights"]]
        self.assertNotIn("strongest-trend", ids)
        naive = {"insight_id": "strongest-trend", "metrics": {"value_column": "Other"}}
        covered = {"insight_id": "strongest-trend", "metrics": {"value_column": "Revenue"}}
        ok = {"status": "ok", "series": [{"column": "Revenue"}]}
        self.assertEqual(_without_superseded_trends([naive, covered], ok), [naive])
        self.assertEqual(_without_superseded_trends([covered], {"status": "insufficient_data"}), [covered])

    def test_brief_leads_with_the_largest_reliable_trend(self):
        priority = self.dashboard["decision_brief"]["priorities"][0]
        self.assertEqual(priority["category"], "validated-trend")
        self.assertEqual(priority["columns"], ["Revenue"])
        self.assertEqual(priority["source"], "temporal_analysis")
        plain = explain_priority(priority, self.dashboard)
        self.assertIn("“Revenue” has risen by about", plain["headline"])
        self.assertIsNone(explain_priority({"category": "validated-trend", "columns": ["Missing"]}, self.dashboard))

    def test_brief_respects_focus_and_exclusions(self):
        temporal = {
            "status": "ok",
            "series": [
                series_item(column="Revenue"),
                series_item(column="Orders", change_over_period={"relative_change": 0.9, "relative_change_ci": [0.5, 1.2]}),
                series_item(column="Visits", trend={"direction": "no_trend", "significant": False}),
            ],
        }
        brief = build_decision_brief({"temporal_analysis": temporal}, READY)
        self.assertEqual(brief["priorities"][0]["columns"], ["Orders"])
        focused = build_decision_brief(
            {"temporal_analysis": temporal, "analysis_context": {"focus_columns": ["Revenue"]}}, READY
        )
        self.assertEqual(focused["priorities"][0]["columns"], ["Revenue"])
        excluded = build_decision_brief(
            {"temporal_analysis": temporal, "analysis_context": {"excluded_columns": ["Orders", "Revenue"]}}, READY
        )
        self.assertNotIn("validated-trend", [item["category"] for item in excluded["priorities"]])

    def test_audit_manifest_records_temporal_status(self):
        frame = dated_frame()
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            frame.to_csv(source, index=False)
            manifest = build_audit_manifest(
                dataset_path=source, dataframe=frame, source_row_count=len(frame),
                truncated=False, source_name="data.csv", prompt="", demo_dataset="",
                run_id=None, viz_spec={}, dashboard_data=self.dashboard,
                engine_version="test", retention_hours=24,
            )
        temporal = manifest["evidence"]["temporal_analysis"]
        self.assertEqual(temporal["contract"], "temporal-analysis-v1")
        self.assertEqual(temporal["status"], "ok")
        self.assertEqual(temporal["series_count"], len(self.dashboard["temporal_analysis"]["series"]))
        self.assertGreaterEqual(temporal["reliable_trend_count"], 1)


class PlainLanguageTrendTests(unittest.TestCase):
    def test_rising_trend_reports_size_range_and_period(self):
        plain = explain_temporal_series(series_item())
        self.assertEqual(plain["headline"], "“Revenue” has risen by about 25% over the analyzed period.")
        self.assertIn("between +18% and +33%", plain["meaning"])
        self.assertIn("the week of 8 Jan 2024 to the week of 16 Dec 2024", plain["meaning"])
        self.assertIn("not a forecast", plain["meaning"])

    def test_falling_trend_and_signed_measure_without_percentage(self):
        falling = explain_temporal_series(
            series_item(
                grain="month",
                first_period="2023-02-01",
                last_period="2025-01-01",
                change_over_period={"relative_change": -0.12, "relative_change_ci": [-0.2, -0.05]},
                trend={"direction": "decreasing", "significant": True},
            )
        )
        self.assertEqual(falling["headline"], "“Revenue” has fallen by about 12% over the analyzed period.")
        self.assertIn("from Feb 2023 to Jan 2025", falling["meaning"])
        signed = explain_temporal_series(series_item(change_over_period={"relative_change": None}))
        self.assertEqual(signed["headline"], "“Revenue” has risen consistently over the analyzed period.")

    def test_no_trend_wording_distinguishes_flat_from_uncertain(self):
        flat = explain_temporal_series(
            series_item(change_over_period={"relative_change": 0.01}, trend={"direction": "no_trend", "significant": False})
        )
        self.assertEqual(flat["headline"], "No reliable trend in “Revenue”.")
        self.assertIn("roughly flat", flat["meaning"])
        uncertain = explain_temporal_series(
            series_item(
                change_over_period={"relative_change": 0.4, "relative_change_ci": [-0.1, 0.9]},
                trend={"direction": "no_trend", "significant": False},
            )
        )
        self.assertIn("could still be chance", uncertain["meaning"])
        self.assertIn("−10% to +90%", uncertain["meaning"])

    def test_incomplete_results_fall_back(self):
        self.assertIsNone(explain_temporal_series(series_item(grain="quarter")))
        self.assertIsNone(explain_temporal_series(series_item(trend={})))
        self.assertIsNone(explain_temporal_series(None))
        self.assertIsNone(explain_temporal({"status": "not_applicable"}))

    def test_summary_and_statistics_wording(self):
        summary = explain_temporal({"status": "ok", "series": [series_item(), series_item(trend={"significant": False})]})
        self.assertEqual(summary["headline"], "We checked 2 measures over time; 1 shows a reliable trend.")
        self.assertIn("not enough history", explain_temporal({"status": "insufficient_data"})["headline"])
        withheld = {"status": "inference_withheld", "reason": "temporal_design_unverified"}
        self.assertIn("Trends over time", explain_statistics(withheld, temporal_checked=True)["meaning"])
        self.assertNotIn("Trends over time", explain_statistics(withheld)["meaning"])


class ReportAndPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = create_saas_growth_demo()
        spec = create_saas_demo_viz_spec()
        cls.spec = spec
        cls.temporal = generate_dashboard_data(frame, spec)
        cls.cross_sectional = generate_dashboard_data(frame.drop(columns=["Date"]), spec)
        cls.app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
        for endpoint in ["analysis_history", "decision_cases", "export", "audit_report", "index"]:
            cls.app.add_url_rule("/" + endpoint, endpoint="vibedash." + endpoint, view_func=lambda: "")

    def render_report(self, dashboard):
        report = build_audit_report_context({"dashboard_data": dashboard})
        with self.app.test_request_context("/"):
            return report, render_template("vibedash_audit_report.html", report=report, report_css="")

    def test_report_adds_trend_section_for_dated_data(self):
        report, html = self.render_report(self.temporal)
        section = report["temporal"]
        self.assertTrue(section["shown"])
        self.assertTrue(section["series"][0]["significant"])
        self.assertTrue(all(item["plain"] for item in section["series"]))
        self.assertIn("4. Trends over time", html)
        self.assertTrue(report["drivers"]["shown"])
        self.assertIn("5. What changed recently, and where", html)
        self.assertIn("6. Statistical validation", html)
        self.assertIn("adjusted p &lt; 0.0001", html)

    def test_report_without_dates_keeps_previous_layout(self):
        report, html = self.render_report(self.cross_sectional)
        self.assertFalse(report["temporal"]["shown"])
        self.assertNotIn("Trends over time", html)
        self.assertIn("4. Statistical validation", html)

    def test_trend_names_are_escaped_in_report_and_page(self):
        dashboard = dict(self.temporal)
        temporal = json.loads(json.dumps(dashboard["temporal_analysis"]))
        temporal["series"][0]["column"] = "<b>Revenue</b>"
        dashboard["temporal_analysis"] = temporal
        _, html = self.render_report(dashboard)
        self.assertIn("&lt;b&gt;Revenue&lt;/b&gt;", html)
        self.assertNotIn("<b>Revenue</b>", html)
        with self.app.test_request_context("/"):
            page = render_template(
                "vibedash_evidence.html", dashboard_data=dashboard, viz_spec=self.spec,
                filename="t.csv", prompt="", session_id="s", export_mode=False,
            )
        self.assertIn("Trends over time", page)
        self.assertIn("Reliable increase", page)
        self.assertIn("&lt;b&gt;Revenue&lt;/b&gt;", page)
        self.assertNotIn("<b>Revenue</b>", page)

    def test_page_explains_insufficient_history(self):
        dashboard = dict(self.temporal)
        dashboard["temporal_analysis"] = {
            "status": "insufficient_data",
            "summary": "No measure has enough complete, regular periods for temporal analysis.",
            "skipped": [{"column": "Revenue", "reason": "too_short", "message": "At most 5 complete days are available."}],
        }
        with self.app.test_request_context("/"):
            page = render_template(
                "vibedash_evidence.html", dashboard_data=dashboard, viz_spec=self.spec,
                filename="t.csv", prompt="", session_id="s", export_mode=False,
            )
        self.assertIn("Not enough history for trend tests", page)


if __name__ == "__main__":
    unittest.main()
