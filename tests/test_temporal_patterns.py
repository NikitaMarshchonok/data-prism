import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template

from vibedash.audit_report import build_audit_report_context
from vibedash.decision_brief import build_decision_brief
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.plain_language import (
    explain_last_period,
    explain_level_shift,
    explain_priority,
    explain_seasonality,
    explain_temporal_patterns,
)
from vibedash.spec import VizSpec
from vibedash.temporal_engine import prepare_series
from vibedash.temporal_patterns import last_period_check, level_shift_test, seasonality_test
from vibedash.temporal_quality import PATTERN_SCENARIOS, evaluate_pattern_detectors


ROOT = Path(__file__).resolve().parents[1]
READY = {"status": "ready", "status_label": "Ready", "checks": []}
WEEKDAY_EFFECT = np.array([0.0, 0.03, 0.05, 0.06, 0.10, -0.12, -0.12])


def store_frame(seed=3, last_week_factor=1.0, days=364):
    """Daily store data: weekday rhythm, mild growth, optional bad last week."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    day = np.arange(days)
    revenue = 1000 * (1 + WEEKDAY_EFFECT[dates.day_of_week]) * (1 + 0.0008 * day) * (1 + rng.normal(0, 0.08, days))
    revenue[-7:] *= last_week_factor
    return pd.DataFrame({"Date": dates, "Revenue": revenue, "Visits": 5000 + rng.normal(0, 300, days)})


def weekly_starts(n):
    return list(pd.date_range("2023-01-02", periods=n, freq="W-MON"))


class SeasonalityTests(unittest.TestCase):
    def test_weekday_rhythm_is_found_with_peak_and_trough(self):
        frame = store_frame()
        daily = prepare_series(frame["Date"], frame["Revenue"], column="Revenue", unit="currency", grain="day")
        result = seasonality_test(daily.values, daily.period_starts, cycle="weekday")
        self.assertLess(result["p_value"], 0.01)
        self.assertEqual(result["cycles_used"], 52)
        self.assertEqual(result["peak"]["label"], "Friday")
        self.assertIn(result["trough"]["label"], {"Saturday", "Sunday"})
        self.assertAlmostEqual(result["amplitude"], 0.22, delta=0.05)
        self.assertEqual(len(result["profile"]), 7)

    def test_noise_profile_is_shrunk_towards_zero(self):
        rng = np.random.default_rng(4)
        starts = list(pd.date_range("2024-01-01", periods=70, freq="D"))
        result = seasonality_test(100 + rng.normal(0, 10, 70), starts, cycle="weekday")
        self.assertGreater(result["p_value"], 0.05)
        self.assertLess(result["amplitude"], 0.10)

    def test_yearly_rhythm_on_monthly_values(self):
        starts = list(pd.date_range("2021-01-01", periods=36, freq="MS"))
        effect = 0.15 * np.cos(2 * np.pi * (np.arange(36) - 11) / 12)  # December peak
        values = 100 * (1 + effect) + np.random.default_rng(5).normal(0, 2, 36)
        result = seasonality_test(values, starts, cycle="month")
        self.assertLess(result["p_value"], 0.01)
        self.assertEqual(result["peak"]["label"], "December")
        self.assertEqual(result["trough"]["label"], "June")

    def test_too_few_cycles_or_non_positive_level_is_not_assessed(self):
        starts = list(pd.date_range("2024-01-01", periods=35, freq="D"))
        self.assertIsNone(seasonality_test(np.ones(35) + np.arange(35), starts, cycle="weekday"))
        starts = list(pd.date_range("2024-01-01", periods=70, freq="D"))
        self.assertIsNone(seasonality_test(np.linspace(-10, -1, 70), starts, cycle="weekday"))
        with self.assertRaises(ValueError):
            seasonality_test(np.ones(70), starts, cycle="quarter")

    def test_result_is_deterministic(self):
        frame = store_frame()
        daily = prepare_series(frame["Date"], frame["Revenue"], column="Revenue", unit="currency", grain="day")
        first = seasonality_test(daily.values, daily.period_starts, cycle="weekday")
        self.assertEqual(first, seasonality_test(daily.values, daily.period_starts, cycle="weekday"))


class LevelShiftTests(unittest.TestCase):
    def test_clear_drop_is_located_and_sized(self):
        rng = np.random.default_rng(6)
        values = 100 - 20 * (np.arange(80) >= 44) + rng.normal(0, 3, 80)
        result = level_shift_test(values, weekly_starts(80))
        self.assertLess(result["p_value"], 0.01)
        self.assertEqual(result["change_period"], weekly_starts(80)[44].date().isoformat())
        self.assertAlmostEqual(result["relative_shift"], -0.20, delta=0.04)
        self.assertEqual(result["periods_before"] + result["periods_after"], 80)

    def test_smooth_growth_is_not_a_shift(self):
        rng = np.random.default_rng(7)
        values = 100 * 1.02 ** np.arange(80) * (1 + rng.normal(0, 0.05, 80))
        self.assertGreater(level_shift_test(values, weekly_starts(80))["p_value"], 0.05)

    def test_signed_series_has_no_percentage(self):
        rng = np.random.default_rng(8)
        values = -10 + 30 * (np.arange(60) >= 30) + rng.normal(0, 3, 60)
        result = level_shift_test(values, weekly_starts(60))
        self.assertLess(result["p_value"], 0.05)
        self.assertGreater(result["absolute_shift"], 20)

    def test_short_or_constant_series_is_not_assessed(self):
        self.assertIsNone(level_shift_test(np.arange(20.0), weekly_starts(20)))
        self.assertIsNone(level_shift_test(np.ones(40), weekly_starts(40)))


class LastPeriodTests(unittest.TestCase):
    def test_sharp_drop_in_the_latest_period_is_flagged(self):
        rng = np.random.default_rng(9)
        values = np.r_[100 + rng.normal(0, 5, 40), 60.0]
        result = last_period_check(values, weekly_starts(41))
        self.assertLess(result["p_value"], 0.01)
        self.assertLess(result["robust_z"], -5)
        self.assertAlmostEqual(result["relative_deviation"], -0.40, delta=0.06)
        self.assertEqual(result["period"], weekly_starts(41)[-1].date().isoformat())

    def test_normal_latest_period_is_not_flagged(self):
        rng = np.random.default_rng(10)
        values = 100 + rng.normal(0, 5, 41)
        self.assertGreater(last_period_check(values, weekly_starts(41))["p_value"], 0.05)

    def test_short_or_flat_history_is_not_assessed(self):
        self.assertIsNone(last_period_check(np.arange(10.0), weekly_starts(10)))
        self.assertIsNone(last_period_check(np.ones(30), weekly_starts(30)))


class PatternQualityGateTests(unittest.TestCase):
    """Ground-truth contract for the three checks; thresholds are the specification."""

    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_pattern_detectors(seeds=300)

    def test_every_scenario_meets_its_threshold(self):
        self.assertEqual(len(self.report["scenarios"]), len(PATTERN_SCENARIOS))
        for scenario in self.report["scenarios"]:
            with self.subTest(scenario=scenario["name"]):
                self.assertTrue(scenario["passed"], scenario)

    def test_every_check_has_null_and_effect_scenarios(self):
        by_check = {}
        for scenario in PATTERN_SCENARIOS:
            by_check.setdefault(scenario.check, set()).add(scenario.has_effect)
        self.assertEqual(set(by_check), {"weekday", "month", "level_shift", "latest_period"})
        self.assertTrue(all(kinds == {True, False} for kinds in by_check.values()))
        with self.assertRaises(ValueError):
            evaluate_pattern_detectors(seeds=0)


class EngineAndProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = generate_dashboard_data(store_frame(last_week_factor=0.6), VizSpec(title="Store"))
        cls.series = {item["column"]: item for item in cls.dashboard["temporal_analysis"]["series"]}

    def test_engine_attaches_corrected_checks_to_each_measure(self):
        json.dumps(self.dashboard["temporal_analysis"], allow_nan=False)
        revenue = self.series["Revenue"]
        self.assertTrue(revenue["seasonality"]["significant"])
        self.assertEqual(revenue["seasonality"]["cycle"], "weekday")
        self.assertTrue(revenue["latest_period"]["significant"])
        for item in self.series.values():
            for key in ("seasonality", "level_shift", "latest_period"):
                with self.subTest(column=item["column"], check=key):
                    self.assertGreaterEqual(item[key]["adjusted_p_value"], item[key]["p_value"])
        # The check must not overwrite the series' own date range.
        self.assertEqual(revenue["last_period"], "2024-12-23")
        self.assertEqual(revenue["latest_period"]["period"], "2024-12-23")
        visits = self.series["Visits"]
        self.assertFalse(visits["seasonality"]["significant"])
        self.assertFalse(visits["latest_period"]["significant"])

    def test_brief_leads_with_the_unusual_latest_week(self):
        priorities = self.dashboard["decision_brief"]["priorities"]
        self.assertEqual(priorities[0]["category"], "last-period-alert")
        self.assertEqual(priorities[0]["columns"], ["Revenue"])
        plain = explain_priority(priorities[0], self.dashboard)
        self.assertIn("“Revenue” (the week of 23 Dec 2024) was about", plain["headline"])
        self.assertIn("below what recent weeks suggested", plain["headline"])
        self.assertIn("missing records", plain["meaning"])

    def test_brief_caps_time_series_priorities(self):
        item = {
            "column": "Sales",
            "grain": "week",
            "trend": {"significant": True, "direction": "increasing", "adjusted_p_value": 0.001},
            "change_over_period": {"relative_change": 0.3, "relative_change_ci": [0.2, 0.4]},
            "level_shift": {"significant": True, "relative_shift": -0.2, "change_period": "2024-05-06"},
            "latest_period": {"significant": True, "relative_deviation": -0.3, "period": "2024-12-23", "adjusted_p_value": 0.01},
        }
        brief = build_decision_brief({"temporal_analysis": {"status": "ok", "series": [item]}}, READY)
        categories = [priority["category"] for priority in brief["priorities"]]
        self.assertEqual(categories[:2], ["last-period-alert", "validated-trend"])
        self.assertNotIn("level-shift", categories)
        shift_only = dict(item, trend={"significant": False}, latest_period={"significant": False})
        brief = build_decision_brief({"temporal_analysis": {"status": "ok", "series": [shift_only]}}, READY)
        self.assertEqual(brief["priorities"][0]["category"], "level-shift")
        plain = explain_priority(brief["priorities"][0], {"temporal_analysis": {"status": "ok", "series": [shift_only]}})
        self.assertEqual(plain["headline"], "“Sales” shifted down by about 20% around the week of 6 May 2024.")

    def test_report_shows_patterns_in_plain_language(self):
        report = build_audit_report_context({"dashboard_data": self.dashboard})
        revenue = next(item for item in report["temporal"]["series"] if item["column"] == "Revenue")
        headlines = [pattern["headline"] for pattern in revenue["patterns"]]
        self.assertTrue(headlines[0].startswith("The latest week of “Revenue”"))
        self.assertIn("“Revenue” follows a weekly rhythm: highest on Fridays", headlines[1])
        self.assertEqual(len(revenue["pattern_details"]), 3)
        app = Flask(__name__, template_folder=str(ROOT / "templates"))
        with app.test_request_context("/"):
            html = render_template("vibedash_audit_report.html", report=report, report_css="")
        self.assertIn("follows a weekly rhythm", html)
        self.assertIn("periods from 2024-01-01 to 2024-12-23.", html)
        self.assertIn("from the week of 1 Jan 2024 to the week of 23 Dec 2024", html)
        self.assertIn("Weekly pattern: peak-to-trough", html)

    def test_evidence_page_lists_each_check(self):
        app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
        for endpoint in ["analysis_history", "decision_cases", "export", "audit_report", "index"]:
            app.add_url_rule("/" + endpoint, endpoint="vibedash." + endpoint, view_func=lambda: "")
        with app.test_request_context("/"):
            page = render_template(
                "vibedash_evidence.html", dashboard_data=self.dashboard, viz_spec=VizSpec(title="Store"),
                filename="t.csv", prompt="", session_id="s", export_mode=False,
            )
        self.assertIn("Weekly pattern:", page)
        self.assertIn("Unusual latest week", page)
        self.assertIn("within normal range", page)


class PlainPatternTests(unittest.TestCase):
    def test_yearly_rhythm_with_two_years_carries_a_caveat(self):
        item = {
            "column": "Sales",
            "grain": "month",
            "seasonality": {
                "significant": True, "cycle": "month", "cycles_used": 2,
                "peak": {"label": "December", "relative": 0.18}, "trough": {"label": "February", "relative": -0.1},
            },
        }
        plain = explain_seasonality(item)
        self.assertEqual(plain["headline"], "“Sales” follows a yearly rhythm: highest in December, lowest in February.")
        self.assertIn("+18%", plain["meaning"])
        self.assertIn("Two years is the minimum", plain["meaning"])

    def test_insignificant_or_incomplete_checks_are_silent(self):
        base = {"column": "Sales", "grain": "week"}
        self.assertIsNone(explain_seasonality(dict(base, seasonality={"significant": False})))
        self.assertIsNone(explain_level_shift(dict(base, level_shift={"significant": True, "change_period": "bad"})))
        self.assertIsNone(explain_last_period(dict(base, latest_period={"significant": True, "period": "2024-01-01"})))
        self.assertEqual(explain_temporal_patterns(base), [])

    def test_spike_wording(self):
        plain = explain_last_period(
            {
                "column": "Orders", "grain": "day",
                "latest_period": {"significant": True, "period": "2024-03-05", "robust_z": 6.0, "relative_deviation": 0.5},
            }
        )
        self.assertEqual(plain["headline"], "The latest day of “Orders” (5 Mar 2024) was about 50% above what recent days suggested.")
        self.assertIn("duplicate import", plain["meaning"])


if __name__ == "__main__":
    unittest.main()
