"""Temporal forecasting oracles and negative contracts, not metric snapshots."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecasting import ForecastTask, _forecast, forecast_series
from src.ml_predictor import predict_target, _select_target


def frame(values=None, size=160):
    if values is None:
        values = 100 + np.arange(size, dtype=float) * .7
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(values)),
                         "demand": values})


TASK = ForecastTask("demand", "date", horizon=7, seasonal_period=7)


class ForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trend = forecast_series(frame(), TASK)

    def test_missing_explicit_target_never_falls_back_or_trains(self):
        data = pd.DataFrame({"feature": range(40), "outcome": np.arange(40) * .5})
        with patch("src.ml_predictor._split_data") as split:
            report = predict_target(data, "typo")
        split.assert_not_called()
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["target_col"], "typo")
        self.assertIsNone(_select_target(data, "typo"))
        self.assertIsNone(_select_target(data, ""))

    def test_dated_legacy_prediction_requires_explicit_design(self):
        with patch("src.ml_predictor._split_data") as split:
            result = predict_target(frame(), "demand")
        split.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertIn("forecast_task", result["metric"])
        independent = frame().assign(feature=np.random.default_rng(45).normal(size=160))
        with patch("src.ml_predictor._split_data", side_effect=ValueError("oracle")) as split:
            predict_target(independent, "demand", sampling_design="independent_rows")
            split.assert_called_once()

    def test_common_predict_entry_point_routes_forecast_and_rejects_conflicts(self):
        with patch("src.forecasting.forecast_series", return_value={"status": "oracle"}) as forecast:
            result = predict_target(frame(), forecast_task=TASK)
            forecast.assert_called_once()
            self.assertEqual(result, {"status": "oracle"})
            forecast.reset_mock()
            for extra in ({"target_column": "other"}, {"sampling_design": "independent_rows"}):
                report = predict_target(frame(), forecast_task=TASK, **extra)
                self.assertEqual(report["status"], "error")
            forecast.assert_not_called()

    def test_naive_forecast_oracles_do_not_use_actual_horizon_values(self):
        history = np.asarray([10, 20, 30, 40, 50, 60, 70], dtype=float)
        self.assertEqual(_forecast(history, 10, 7, "last_value").tolist(), [70] * 10)
        self.assertEqual(_forecast(history, 10, 7, "seasonal_naive").tolist(),
                         [10, 20, 30, 40, 50, 60, 70, 10, 20, 30])

    def test_temporal_folds_are_ordered_and_final_horizon_is_reserved(self):
        report = self.trend
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["cv_folds"], 3)
        self.assertEqual(report["test_rows"], 7)
        previous_end = None
        for fold in report["validation_folds"]:
            self.assertLess(fold["train_end"], fold["validation_start"])
            self.assertLess(fold["validation_end"], report["holdout"]["start"])
            if previous_end:
                self.assertLess(previous_end, fold["validation_start"])
            previous_end = fold["validation_end"]
        self.assertLess(report["holdout"]["train_end"], report["holdout"]["start"])
        self.assertEqual(sum(row["selected"] for row in report["model_comparison"]), 1)
        json.dumps(report, allow_nan=False)

    def test_known_trend_beats_seasonal_baseline_without_significance_claim(self):
        report = self.trend
        self.assertEqual(report["model_name"], "lag_ridge")
        self.assertGreater(report["holdout"]["mae_improvement"], .8)
        self.assertEqual(report["predictive_support"]["statistical_significance"], "not_assessed")
        self.assertEqual(report["predictive_support"]["production_readiness"], "not_established")
        self.assertIsNone(report["prediction_intervals"])

    def test_perfect_seasonal_baseline_is_retained_not_replaced_by_complexity(self):
        periodic = frame(np.tile([10., 20., 30., 70., 60., 50., 40.], 24))
        report = forecast_series(periodic, TASK)
        self.assertEqual(report["model_name"], "seasonal_naive")
        self.assertEqual(report["holdout"]["metrics"]["mae"], 0)
        self.assertIsNone(report["holdout"]["mae_improvement"])
        self.assertEqual(report["predictive_support"]["status"], "baseline_selected")

    def test_mutating_holdout_cannot_change_selection_cv_or_holdout_predictions(self):
        data = frame()
        changed = data.copy()
        changed.loc[changed.index[-7:], "demand"] += 10000
        report = forecast_series(changed, TASK)
        self.assertEqual(report["model_name"], self.trend["model_name"])
        self.assertEqual(report["baseline_name"], self.trend["baseline_name"])
        self.assertEqual(report["model_comparison"], self.trend["model_comparison"])
        for before, after in zip(self.trend["holdout"]["predictions"], report["holdout"]["predictions"]):
            self.assertEqual(before["prediction"], after["prediction"])
            self.assertEqual(before["baseline"], after["baseline"])
        self.assertNotEqual(report["holdout"]["metrics"], self.trend["holdout"]["metrics"])

    def test_target_components_and_future_weather_are_never_predictors(self):
        data = frame()
        data["casual"] = data["demand"] * .3
        data["registered"] = data["demand"] * .7
        data["future_weather"] = data["demand"].shift(-7)
        report = forecast_series(data, TASK)
        self.assertEqual(report["model_comparison"], self.trend["model_comparison"])
        self.assertEqual(report["holdout"], self.trend["holdout"])
        self.assertEqual(report["future_forecast"], self.trend["future_forecast"])
        self.assertEqual(report["unused_columns"], ["casual", "registered", "future_weather"])

    def test_row_order_and_index_do_not_change_temporal_evaluation(self):
        data = frame().sample(frac=1, random_state=2)
        data.index = [1] * len(data)
        report = forecast_series(data, TASK)
        self.assertEqual(report, self.trend)

    def test_full_horizon_calls_only_receive_prefixes_and_refit_is_separate(self):
        from src.forecasting import _forecast as original
        lengths = []
        def observe(history, *args):
            lengths.append(len(history))
            return original(history, *args)
        with patch("src.forecasting._forecast", side_effect=observe):
            report = forecast_series(frame(), TASK)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(lengths[:12], [132, 139, 146] * 4)
        self.assertEqual(lengths[12:14], [153, 153])
        self.assertEqual(lengths[-1], 160)
        self.assertEqual(len(report["future_forecast"]), 7)
        self.assertGreater(report["future_forecast"][0]["date"], report["future_fit_through"])

    def test_failing_candidate_is_visible_and_never_selectable(self):
        from src.forecasting import _forecast as original
        def failing(history, horizon, period, candidate):
            if candidate == "lag_ridge":
                raise ValueError("synthetic failure")
            return original(history, horizon, period, candidate)
        with patch("src.forecasting._forecast", side_effect=failing):
            report = forecast_series(frame(), TASK)
        self.assertEqual(report["status"], "ok")
        failed = next(row for row in report["model_comparison"] if row["model_name"] == "lag_ridge")
        self.assertEqual(failed["status"], "error")
        self.assertFalse(failed["selected"])
        self.assertIsNone(failed["mean_mae"])

    def test_scaler_fits_only_past_prefixes_until_separate_future_refit(self):
        from sklearn.preprocessing import StandardScaler
        original_fit = StandardScaler.fit
        maxima = []
        def audit_fit(scaler, X, y=None, **kwargs):
            maxima.append(float(np.max(X)))
            return original_fit(scaler, X, y, **kwargs)
        data = frame()
        data.loc[data.index[-7:], "demand"] = 1_000_000
        with patch.object(StandardScaler, "fit", audit_fit):
            result = forecast_series(data, TASK)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(maxima), 5)  # 3 folds, locked holdout, future refit
        self.assertTrue(all(value < 1000 for value in maxima[:-1]))
        self.assertEqual(maxima[-1], 1_000_000)

    def test_extreme_overflow_does_not_produce_nonfinite_json_metrics(self):
        data = frame(np.tile([1e308, -1e308], 80))
        result = forecast_series(data, TASK)
        self.assertEqual(result["status"], "error")
        json.dumps(result, allow_nan=False)

    def test_daily_iso_strings_and_datetime_inputs_have_identical_meaning(self):
        data = frame()
        data["date"] = data.date.dt.strftime("%Y-%m-%d")
        self.assertEqual(forecast_series(data, TASK), self.trend)

    def test_task_validation_rejects_implicit_or_unknown_settings(self):
        tasks = [None, {}, {"target_column": "demand", "time_column": "date", "horizon": True},
                 {"target_column": "demand", "time_column": "date", "horizon": 0},
                 {"target_column": "demand", "time_column": "date", "horizon": 91},
                 {"target_column": "demand", "time_column": "date", "horizon": 7, "seasonal_period": 0},
                 {"target_column": "demand", "time_column": "date", "horizon": 7, "shuffle": True},
                 {"target_column": "date", "time_column": "date", "horizon": 7}]
        for task in tasks:
            with self.subTest(task=task):
                result = forecast_series(frame(), task)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["future_forecast"], [])

    def test_bad_data_never_imputed_aggregated_or_silently_substituted(self):
        base = frame()
        invalid = [base.drop(columns="demand"), base.drop(columns="date"), base.iloc[:40],
                   base.drop(index=20), pd.concat([base, base.iloc[:1]]),
                   base.assign(demand=base.demand.astype(str)), base.assign(demand=True),
                   base.assign(date=np.arange(len(base))),
                   base.assign(date=base.date + pd.Timedelta(hours=1)),
                   base.assign(date=base.date.dt.tz_localize("UTC")),
                   base.assign(demand=np.arange(len(base), dtype=np.int64) + 2 ** 53)]
        for value in (np.nan, np.inf, -np.inf):
            changed = base.copy()
            changed.loc[20, "demand"] = value
            invalid.append(changed)
        duplicate_columns = base.copy()
        duplicate_columns.columns = ["date", "date"]
        invalid.append(duplicate_columns)
        for data in invalid:
            with self.subTest(dtypes=str(data.dtypes)), patch("src.forecasting._forecast") as forecast:
                report = forecast_series(data, TASK)
                self.assertEqual(report["status"], "error")
                forecast.assert_not_called()
                json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
