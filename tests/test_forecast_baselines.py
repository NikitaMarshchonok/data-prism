"""Historical constants are genuine prefix-only baselines, not learned signals."""
import json
import math
import statistics
import unittest
from unittest.mock import patch

import numpy as np

from src.forecasting import (
    BASELINES, CANDIDATES, CANDIDATE_SUITE, LEGACY_CANDIDATES, LEGACY_CANDIDATE_SUITE,
    ForecastTask, _forecast, forecast_series,
)
from tests.test_forecasting import frame, TASK
from tests.test_forecast_policy import POLICY
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from vibedash.forecast_tasks import forecast_task_from_form, forecast_audit_view, run_confirmed_forecast


class ForecastHistoricalBaselineTests(unittest.TestCase):
    def test_independent_constant_oracles_include_odd_even_and_outlier_histories(self):
        for history, mean, median in (([1., 2., 3., 100.], 26.5, 2.5),
                                      ([0., 0., 90.], 30., 0.), ([-9., -3., 6.], -2., -3.)):
            with self.subTest(history=history):
                np.testing.assert_array_equal(_forecast(np.array(history), 12, 7, 'historical_mean'), [mean] * 12)
                np.testing.assert_array_equal(_forecast(np.array(history), 12, 7, 'historical_median'), [median] * 12)
        for name in ('historical_mean', 'historical_median'):
            result = _forecast(np.full(100, 1e308), 3, 7, name)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result, [1e308] * 3)

    def test_simple_mean_can_be_selected_and_is_not_promoted_as_learned_support(self):
        data = frame(np.tile([0., 2.], 80))
        with patch('src.forecasting.CANDIDATES', BASELINES):
            report = forecast_series(data, TASK)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['model_name'], 'historical_mean')
        self.assertEqual(report['baseline_name'], 'historical_mean')
        self.assertEqual(report['predictive_support']['status'], 'baseline_selected')
        self.assertEqual(report['holdout']['metrics'], report['holdout']['baseline_metrics'])
        self.assertEqual(report['holdout']['mae_improvement'], 0.)
        self.assertEqual([row['prediction'] for row in report['future_forecast']], [1.] * 7)

    def test_sparse_series_can_retain_zero_median_instead_of_fitting_spikes(self):
        values = np.zeros(160)
        values[131], values[159] = 100., 200.
        report = forecast_series(frame(values), TASK)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['model_name'], 'historical_median')
        self.assertEqual(report['baseline_name'], 'historical_median')
        self.assertEqual(report['predictive_support']['status'], 'baseline_selected')
        self.assertEqual([row['prediction'] for row in report['future_forecast']], [0.] * 7)
        self.assertIsNone(report['prediction_intervals'])

    def test_delayed_validation_and_holdout_match_independent_prefix_oracles(self):
        values = np.array([100 + (i * i % 113) for i in range(160)], dtype=float)
        h, delay = 3, 10
        task = ForecastTask('demand', 'date', h, 7, delay)
        with patch('src.forecasting.CANDIDATES', BASELINES):
            report = forecast_series(frame(values), task)
        self.assertEqual(report['status'], 'ok')
        end, start = len(values) - delay, len(values) - delay - h
        origins = [start - delay - h * k for k in (3, 2, 1)]
        functions = {'historical_mean': statistics.mean, 'historical_median': statistics.median}
        for name, function in functions.items():
            row = next(row for row in report['model_comparison'] if row['model_name'] == name)
            for origin, score in zip(origins, row['fold_metrics']):
                constant = function(values[:origin - delay].tolist())
                expected = math.fsum(abs(float(value) - constant) for value in values[origin:origin + h]) / h
                self.assertAlmostEqual(score['mae'], expected, places=11)
            with patch('src.forecasting.CANDIDATES', (name,)):
                forced = forecast_series(frame(values), task)
            self.assertEqual(forced['status'], 'ok')
            expected_holdout = function(values[:start - delay].tolist())
            expected_future = function(values[:end].tolist())
            for row in forced['holdout']['predictions']:
                self.assertAlmostEqual(row['prediction'], expected_holdout, places=11)
            for row in forced['future_forecast']:
                self.assertAlmostEqual(row['prediction'], expected_future, places=11)
        # Every selected baseline is checked on the locked holdout as well.
        name = report['model_name']
        if name in functions:
            constant = functions[name](values[:start - delay].tolist())
            for row in report['holdout']['predictions']:
                self.assertAlmostEqual(row['prediction'], constant, places=11)
        self.assertEqual(report['holdout']['end'], frame(values).date.iloc[end - 1].strftime('%Y-%m-%d'))

    def test_new_baseline_validation_cannot_use_holdout_or_unreleased_gap(self):
        task = ForecastTask('demand', 'date', 3, 7, 10)
        data = frame()
        before = forecast_series(data, task)
        # Retrospective cutoff=150, holdout=[147,150), selection ends at137.
        data.loc[137:, 'demand'] += 10000.
        after = forecast_series(data, task)
        self.assertEqual(before['model_comparison'], after['model_comparison'])
        self.assertEqual(before['model_name'], after['model_name'])
        for first, second in zip(before['holdout']['predictions'], after['holdout']['predictions']):
            self.assertEqual(first['prediction'], second['prediction'])
            self.assertEqual(first['baseline'], second['baseline'])
        self.assertNotEqual(before['holdout']['metrics'], after['holdout']['metrics'])

    def test_predeclared_asymmetric_loss_can_select_new_baseline_without_holdout_hindsight(self):
        constants = dict(zip(CANDIDATES, [8., 14., 9.5, 12., 11.5, 8.]))
        def oracle(history, horizon, period, candidate): return np.repeat(constants[candidate], horizon)
        data = frame(np.full(160, 10.))
        changed = data.copy()
        changed.loc[153:, 'demand'] = 100.
        with patch('src.forecasting._forecast', side_effect=oracle):
            unweighted = forecast_series(data, TASK)
            weighted = forecast_series(data, TASK, decision_policy=POLICY)
            after = forecast_series(changed, TASK, decision_policy=POLICY)
        self.assertEqual(unweighted['model_name'], 'historical_mean')
        self.assertEqual(weighted['model_name'], 'lag_ridge')
        self.assertEqual(weighted['baseline_name'], 'historical_median')
        self.assertEqual(weighted['model_comparison'], after['model_comparison'])
        self.assertEqual(weighted['baseline_name'], after['baseline_name'])

    def test_legacy_suite_is_explicit_reproducible_and_unknown_suite_never_trains(self):
        legacy = forecast_series(frame(), TASK, candidate_suite=LEGACY_CANDIDATE_SUITE)
        with patch('src.forecasting.CANDIDATES', LEGACY_CANDIDATES):
            restricted = forecast_series(frame(), TASK)
        for key in ('model_comparison', 'model_name', 'baseline_name', 'holdout', 'future_forecast'):
            self.assertEqual(legacy[key], restricted[key], key)
        self.assertEqual(legacy['baseline_candidates'], list(LEGACY_CANDIDATES[:2]))
        extended = forecast_series(frame(), TASK)
        self.assertEqual(extended['candidate_suite'], CANDIDATE_SUITE)
        self.assertEqual(extended['baseline_candidates'], list(BASELINES))
        self.assertEqual([row['model_name'] for row in extended['model_comparison']], list(CANDIDATES))
        for suite in ('unknown', None, True, [], {}):
            with self.subTest(suite=suite), patch('src.forecasting._forecast') as fit:
                result = forecast_series(frame(), TASK, candidate_suite=suite)
                self.assertEqual(result['status'], 'error')
                fit.assert_not_called()

    def test_web_executor_and_aggregate_audit_record_new_suite_without_raw_predictions(self):
        data = daily_frame()
        semantics = confirmed(data)
        report = run_confirmed_forecast(data, semantics, forecast_task_from_form(data, semantics, FORM))
        audit = forecast_audit_view(report)
        self.assertEqual(audit['candidate_suite'], CANDIDATE_SUITE)
        self.assertEqual(audit['baseline_candidates'], list(BASELINES))
        self.assertIn('point estimates', report['domain_constraints']['note'])
        self.assertNotIn('are expected values', report['domain_constraints']['note'])
        self.assertNotIn('predictions', json.dumps(audit))
        json.dumps(audit, allow_nan=False)


if __name__ == '__main__': unittest.main()
