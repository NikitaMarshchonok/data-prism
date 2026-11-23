import unittest

import numpy as np
import pandas as pd

from src.demo_data import create_saas_growth_demo
from vibedash.temporal_engine import (
    MAX_MEASURES,
    MIN_PERIODS,
    TEMPORAL_CONTRACT,
    TemporalAnalysisEngine,
    TemporalSeriesError,
    _hamed_rao_factor,
    mann_kendall_trend,
    prepare_series,
)
from vibedash.temporal_quality import SCENARIOS, evaluate_trend_detector


def daily_frame(start, periods, values=None):
    dates = pd.date_range(start, periods=periods, freq="D")
    if values is None:
        values = np.arange(periods, dtype=float)
    return pd.Series(dates), pd.Series(values, dtype=float)


class PrepareSeriesTests(unittest.TestCase):
    def test_grain_follows_the_observed_span(self):
        cases = {30: "day", 200: "week", 800: "month"}
        for days, grain in cases.items():
            with self.subTest(days=days):
                times, values = daily_frame("2024-01-01", days)
                prepared = prepare_series(times, values, column="x", unit="number")
                self.assertEqual(prepared.grain, grain)

    def test_partial_edge_months_are_excluded(self):
        # 2023-01-15 .. 2025-03-10: January 2023 and March 2025 are partial.
        times = pd.Series(pd.date_range("2023-01-15", "2025-03-10", freq="D"))
        values = pd.Series(np.ones(len(times)))
        prepared = prepare_series(times, values, column="orders", unit="count")
        self.assertEqual(prepared.grain, "month")
        self.assertEqual(prepared.excluded_edge_periods, ["2023-01", "2025-03"])
        self.assertEqual(prepared.period_starts[0], pd.Timestamp("2023-02-01"))
        self.assertEqual(prepared.period_starts[-1], pd.Timestamp("2025-02-01"))

    def test_complete_edge_months_are_kept(self):
        times = pd.Series(pd.date_range("2023-01-01", "2024-12-31", freq="D"))
        prepared = prepare_series(times, pd.Series(np.ones(len(times))), column="x", unit="count")
        self.assertEqual(prepared.excluded_edge_periods, [])
        self.assertEqual(prepared.period_count, 24)

    def test_partial_edge_weeks_are_excluded(self):
        # 2024-01-03 is a Wednesday; 2024-06-27 is a Thursday.
        times = pd.Series(pd.date_range("2024-01-03", "2024-06-27", freq="D"))
        prepared = prepare_series(times, pd.Series(np.ones(len(times))), column="x", unit="count")
        self.assertEqual(prepared.grain, "week")
        self.assertEqual(len(prepared.excluded_edge_periods), 2)
        self.assertEqual(prepared.period_starts[0].day_name(), "Monday")
        self.assertTrue(np.allclose(prepared.values, 7.0))

    def test_business_day_data_moves_to_a_coarser_grain(self):
        times = pd.Series(pd.bdate_range("2024-01-01", periods=70))
        values = pd.Series(np.linspace(10, 20, len(times)))
        prepared = prepare_series(times, values, column="x", unit="number")
        self.assertEqual(prepared.grain, "week")
        self.assertEqual(prepared.empty_periods, [])

    def test_sum_for_counts_and_currency_mean_for_other_units(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        times = pd.Series(dates.repeat(2))
        values = pd.Series([1.0, 3.0] * 20)
        for unit, expected, aggregation in (
            ("count", 4.0, "sum"),
            ("currency", 4.0, "sum"),
            ("ratio", 2.0, "mean"),
            ("percent", 2.0, "mean"),
            ("number", 2.0, "mean"),
        ):
            with self.subTest(unit=unit):
                prepared = prepare_series(times, values, column="x", unit=unit)
                self.assertEqual(prepared.aggregation, aggregation)
                self.assertTrue(np.allclose(prepared.values, expected))

    def test_monthly_totals_are_tested_as_per_day_rates(self):
        times = pd.Series(pd.date_range("2023-01-01", "2024-12-31", freq="D"))
        prepared = prepare_series(times, pd.Series(np.full(len(times), 5.0)), column="x", unit="currency")
        self.assertTrue(prepared.calendar_adjusted)
        # A constant daily rate must not look like a seasonal pattern.
        self.assertTrue(np.allclose(prepared.values, 5.0))

        averaged = prepare_series(times, pd.Series(np.full(len(times), 5.0)), column="x", unit="ratio")
        self.assertFalse(averaged.calendar_adjusted)

    def test_too_short_and_empty_series_raise_with_a_reason(self):
        times, values = daily_frame("2024-01-01", MIN_PERIODS - 1)
        with self.assertRaises(TemporalSeriesError) as context:
            prepare_series(times, values, column="x", unit="number")
        self.assertEqual(context.exception.reason, "too_short")

        with self.assertRaises(TemporalSeriesError) as context:
            prepare_series(pd.Series(["not a date"] * 20), pd.Series(range(20)), column="x", unit="number")
        self.assertEqual(context.exception.reason, "no_values")

    def test_irregular_sparse_data_is_rejected_instead_of_filled(self):
        # Observations every 45 days over two years: no grain is dense enough.
        times = pd.Series(pd.date_range("2023-01-01", periods=17, freq="45D"))
        with self.assertRaises(TemporalSeriesError) as context:
            prepare_series(times, pd.Series(np.arange(17.0)), column="x", unit="number")
        self.assertIn(context.exception.reason, {"sparse", "too_short"})

    def test_non_finite_values_and_timezones_are_handled(self):
        times = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D", tz="Europe/Berlin"))
        values = pd.Series(np.arange(30.0))
        values.iloc[5] = np.inf
        values.iloc[6] = np.nan
        prepared = prepare_series(times, values, column="x", unit="number")
        self.assertEqual(prepared.grain, "day")
        self.assertTrue(np.isfinite(prepared.values).sum() >= MIN_PERIODS)


class MannKendallTests(unittest.TestCase):
    def test_clear_increasing_trend_is_detected_with_a_covering_interval(self):
        rng = np.random.default_rng(1)
        values = 100 + 2.0 * np.arange(48) + rng.normal(0, 5, 48)
        result = mann_kendall_trend(values)
        self.assertLess(result["p_value"], 0.001)
        self.assertGreater(result["z_statistic"], 0)
        lower, upper = result["slope_ci"]
        self.assertLess(lower, 2.0)
        self.assertGreater(upper, 2.0)
        self.assertAlmostEqual(result["slope_per_period"], 2.0, delta=0.3)

    def test_decreasing_trend_has_negative_slope_and_interval(self):
        values = 500 - 3.0 * np.arange(40) + np.random.default_rng(2).normal(0, 4, 40)
        result = mann_kendall_trend(values)
        self.assertLess(result["slope_per_period"], 0)
        self.assertLess(result["slope_ci"][1], 0)

    def test_constant_series_reports_no_trend(self):
        result = mann_kendall_trend(np.full(20, 7.0))
        self.assertEqual(result["s_statistic"], 0.0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["slope_per_period"], 0.0)

    def test_slope_is_robust_to_a_single_extreme_value(self):
        values = 10 + np.arange(30, dtype=float)
        values[15] = 10_000
        self.assertAlmostEqual(mann_kendall_trend(values)["slope_per_period"], 1.0, places=6)

    def test_short_series_is_refused(self):
        with self.assertRaises(ValueError):
            mann_kendall_trend(np.arange(MIN_PERIODS - 1, dtype=float))

    def test_variance_correction_never_shrinks_the_variance(self):
        rng = np.random.default_rng(3)
        alternating = np.tile([1.0, -1.0], 20) + rng.normal(0, 0.1, 40)
        self.assertEqual(_hamed_rao_factor(alternating, 0.0), 1.0)
        for _ in range(50):
            values = rng.normal(size=30)
            self.assertGreaterEqual(_hamed_rao_factor(values, 0.0), 1.0)

    def test_autocorrelation_widens_the_interval(self):
        rng = np.random.default_rng(4)
        noise = np.empty(60)
        noise[0] = 0.0
        for index in range(1, 60):
            noise[index] = 0.9 * noise[index - 1] + rng.normal()
        result = mann_kendall_trend(noise)
        self.assertGreater(result["variance_inflation"], 2.0)


class EngineTests(unittest.TestCase):
    def frame(self, days=400, slope=0.5, seed=5):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=days, freq="D")
        return pd.DataFrame(
            {
                "Date": dates,
                "Revenue": 1000 + slope * np.arange(days) + rng.normal(0, 20, days),
                "Flat": 50 + rng.normal(0, 5, days),
            }
        )

    def test_reports_reliable_and_absent_trends(self):
        result = TemporalAnalysisEngine(self.frame(), units={"Revenue": "currency"}).analyze()
        self.assertEqual(result["contract"], TEMPORAL_CONTRACT)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["time_column"], "Date")
        by_column = {item["column"]: item for item in result["series"]}
        revenue = by_column["Revenue"]
        self.assertEqual(revenue["aggregation"], "sum")
        self.assertEqual(revenue["grain"], "week")
        self.assertEqual(revenue["trend"]["direction"], "increasing")
        self.assertTrue(revenue["trend"]["significant"])
        self.assertGreater(revenue["trend"]["relative_change_per_year"], 0)
        self.assertEqual(by_column["Flat"]["trend"]["direction"], "no_trend")
        for item in result["series"]:
            trend = item["trend"]
            self.assertGreaterEqual(trend["adjusted_p_value"], trend["p_value"])
        self.assertTrue(result["methodology_notes"])

    def test_not_applicable_without_dates_or_measures(self):
        frame = self.frame()
        self.assertEqual(TemporalAnalysisEngine(frame.drop(columns=["Date"])).analyze()["status"], "not_applicable")
        self.assertEqual(TemporalAnalysisEngine(frame[["Date"]]).analyze()["status"], "not_applicable")
        self.assertEqual(TemporalAnalysisEngine(frame.iloc[0:0]).analyze()["status"], "not_applicable")

    def test_too_short_history_is_insufficient_not_an_error(self):
        result = TemporalAnalysisEngine(self.frame(days=8)).analyze()
        self.assertEqual(result["status"], "insufficient_data")
        self.assertTrue(result["skipped"])
        self.assertEqual(result["skipped"][0]["reason"], "too_short")

    def test_measure_count_is_bounded_and_focus_comes_first(self):
        frame = self.frame()
        rng = np.random.default_rng(6)
        for index in range(8):
            frame[f"M{index}"] = rng.normal(size=len(frame))
        result = TemporalAnalysisEngine(frame, analysis_context={"focus_columns": ["M7"]}).analyze()
        self.assertEqual(len(result["series"]), MAX_MEASURES)
        self.assertEqual(result["series"][0]["column"], "M7")

    def test_excluded_columns_are_respected(self):
        result = TemporalAnalysisEngine(self.frame(), analysis_context={"excluded_columns": ["Revenue"]}).analyze()
        self.assertNotIn("Revenue", [item["column"] for item in result["series"]])

    def test_invalid_arguments(self):
        with self.assertRaises(TypeError):
            TemporalAnalysisEngine([1, 2, 3])
        with self.assertRaises(ValueError):
            TemporalAnalysisEngine(self.frame(), alpha=1.5)

    def test_saas_demo(self):
        result = TemporalAnalysisEngine(create_saas_growth_demo()).analyze()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["series"])
        for item in result["series"]:
            self.assertIn(item["grain"], {"day", "week", "month"})
            self.assertGreaterEqual(item["period_count"], MIN_PERIODS)


class TrendQualityGateTests(unittest.TestCase):
    """Ground-truth contract: false alarms stay near alpha, real trends are found."""

    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_trend_detector(seeds=1000)

    def test_every_scenario_meets_its_threshold(self):
        for scenario in self.report["scenarios"]:
            with self.subTest(scenario=scenario["name"]):
                self.assertTrue(scenario["passed"], scenario)
        self.assertTrue(self.report["passed"])
        self.assertEqual(len(self.report["scenarios"]), len(SCENARIOS))

    def test_autocorrelation_correction_beats_the_naive_baseline(self):
        ar1 = next(item for item in self.report["scenarios"] if item["name"] == "ar1_noise")
        self.assertGreater(ar1["naive_ols_rate"], 2 * ar1["engine_rate"])

    def test_evaluation_is_deterministic(self):
        self.assertEqual(evaluate_trend_detector(seeds=20), evaluate_trend_detector(seeds=20))
        with self.assertRaises(ValueError):
            evaluate_trend_detector(seeds=0)


if __name__ == "__main__":
    unittest.main()
