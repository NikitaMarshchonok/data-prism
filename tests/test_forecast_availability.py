"""Release-delay oracles: fitting, selection and scoring must respect availability."""
import copy
from datetime import date, timedelta
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.forecasting import ForecastTask, _forecast, forecast_series
from tests.test_forecasting import frame, TASK
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from tests.test_forecast_policy import POLICY_FORM
from vibedash.forecast_tasks import (
    AVAILABILITY_TASK_CONTRACT, forecast_task_from_form, run_confirmed_forecast, forecast_audit_view,
)


class ForecastAvailabilityTests(unittest.TestCase):
    def test_invalid_delay_never_trains_or_silently_defaults(self):
        fields = {'target_column': 'demand', 'time_column': 'date', 'horizon': 7}
        for delay in (-1, 31, True, 2.0, '2', None, [], float('nan')):
            with self.subTest(delay=delay), patch('src.forecasting._forecast') as fit:
                result = forecast_series(frame(), fields | {'reporting_delay_days': delay})
                self.assertEqual(result['status'], 'error')
                fit.assert_not_called()

    def test_explicit_zero_delay_preserves_legacy_settings_and_numerical_result(self):
        self.assertEqual(forecast_series(frame(), TASK),
                         forecast_series(frame(), ForecastTask('demand', 'date', 7, 7, 0)))
        report = forecast_series(frame(), TASK)
        self.assertEqual(report['contract'], 'daily-forecast-v1')
        self.assertNotIn('reporting_delay_days', report['task'])

    def test_minimum_history_accounts_for_both_release_embargo_and_unreleased_outcomes(self):
        task = ForecastTask('demand', 'date', 3, 7, 10)
        minimum = 30 + 4 * 3 + 3 * 10
        with patch('src.forecasting.CANDIDATES', ('last_value', 'seasonal_naive')):
            self.assertEqual(forecast_series(frame(size=minimum - 1), task)['status'], 'error')
            report = forecast_series(frame(size=minimum), task)
        self.assertEqual(report['status'], 'ok', report)
        self.assertEqual(report['validation_folds'][0]['train_rows'], 30)

    def test_independent_naive_and_seasonal_oracles_bridge_gap_and_score_only_released_dates(self):
        # Non-periodic data, and delays both below/above horizon and seasonal period.
        values = np.asarray([100 + (i * i % 113) for i in range(160)], dtype=float)
        for delay in (1, 2, 8, 30):
            horizon, period = 3, 7
            task = ForecastTask('demand', 'date', horizon, period, delay)
            with patch('src.forecasting.CANDIDATES', ('last_value', 'seasonal_naive')):
                report = forecast_series(frame(values), task)
            self.assertEqual(report['status'], 'ok')
            end = len(values) - delay
            holdout = end - horizon
            starts = [holdout - delay - horizon * k for k in (3, 2, 1)]
            def oracle(start, name):
                prefix = values[:start - delay].tolist()
                if name == 'last_value': return [prefix[-1]] * horizon
                cycle = prefix[-period:]
                return [cycle[step % period] for step in range(delay, delay + horizon)]
            losses = {name: [sum(abs(float(y) - p) for y, p in zip(values[start:start + horizon],
                       oracle(start, name))) / horizon for start in starts]
                      for name in ('last_value', 'seasonal_naive')}
            winner = min(losses, key=lambda name: sum(losses[name]) / 3)
            self.assertEqual(report['model_name'], winner)
            for model in report['model_comparison']:
                self.assertEqual([item['mae'] for item in model['fold_metrics']], losses[model['model_name']])
            self.assertEqual([row['prediction'] for row in report['holdout']['predictions']], oracle(holdout, winner))
            self.assertEqual([row['actual'] for row in report['holdout']['predictions']], values[holdout:end].tolist())
            self.assertEqual([row['prediction'] for row in report['future_forecast']], oracle(len(values), winner))
            self.assertEqual(report['holdout']['end'], (date(2020, 1, 1) + timedelta(days=end - 1)).isoformat())
            self.assertEqual(report['future_forecast'][0]['date'], '2020-06-09')
            self.assertEqual(len(report['future_forecast']), horizon)
            json.dumps(report, allow_nan=False)

    def test_all_real_model_calls_receive_only_available_prefix_and_bridge_full_delay(self):
        source = frame()
        values = source.demand.to_numpy()
        delay, horizon = 5, 7
        calls = []
        def observe(history, steps, period, candidate):
            calls.append((history.copy(), steps, candidate))
            return _forecast(history, steps, period, candidate)
        with patch('src.forecasting._forecast', side_effect=observe):
            report = forecast_series(source, ForecastTask('demand', 'date', horizon, 7, delay))
        self.assertEqual(report['status'], 'ok')
        self.assertEqual([len(item[0]) for item in calls[:12]], [117, 124, 131] * 4)
        self.assertTrue(all(len(history) == 143 for history, _, _ in calls[12:-1]))
        self.assertEqual(len(calls[-1][0]), 155)
        for history, steps, _ in calls:
            np.testing.assert_array_equal(history, values[:len(history)])
            self.assertEqual(steps, delay + horizon)
        for fold in report['validation_folds'] + [report['holdout']]:
            self.assertEqual((date.fromisoformat(fold['forecast_origin']) - date.fromisoformat(fold['train_end'])).days, delay)
        for fold in report['validation_folds']:
            self.assertLessEqual(fold['validation_end'], report['holdout']['train_end'])
        self.assertEqual(report['future_forecast_origin'], '2020-06-08')
        self.assertEqual(report['future_fit_through'], '2020-06-03')

    def test_unreleased_tail_cannot_change_selection_holdout_scaling_or_future_forecast(self):
        task = ForecastTask('demand', 'date', 3, 7, 30)
        original = frame()
        changed = original.copy()
        changed.loc[changed.index[-30:], 'demand'] = 1_000_000
        # Delay exceeds the total of several scoring horizons: catches leakage
        # via validation outcomes, not just direct training/predictor leakage.
        before = forecast_series(original, task)
        from sklearn.preprocessing import StandardScaler
        original_fit = StandardScaler.fit
        maxima = []
        def audit_fit(scaler, X, y=None, **kwargs):
            maxima.append(float(np.max(X)))
            return original_fit(scaler, X, y, **kwargs)
        with patch.object(StandardScaler, 'fit', audit_fit):
            after = forecast_series(changed, task)
        self.assertEqual(before['status'], 'ok')
        self.assertEqual(after, before)
        self.assertEqual(len(maxima), 5)
        self.assertTrue(all(value < 1000 for value in maxima))

    def test_pre_holdout_unreleased_gap_cannot_influence_selection_or_holdout_evaluation(self):
        task = ForecastTask('demand', 'date', 3, 7, 10)
        source = frame()
        before = forecast_series(source, task)
        changed = source.copy()
        holdout_start = len(source) - 10 - 3
        changed.loc[changed.index[holdout_start - 10:holdout_start], 'demand'] = 1_000_000
        after = forecast_series(changed, task)
        self.assertEqual(after['status'], 'ok')
        self.assertEqual(after['model_comparison'], before['model_comparison'])
        self.assertEqual(after['model_name'], before['model_name'])
        self.assertEqual(after['baseline_name'], before['baseline_name'])
        self.assertEqual(after['holdout'], before['holdout'])
        self.assertEqual(before['availability']['selection_outcomes_available_through'],
                         before['holdout']['train_end'])

    def test_released_holdout_mutation_cannot_reselect_method_or_change_its_predictions(self):
        task = ForecastTask('demand', 'date', 7, 7, 4)
        source = frame()
        before = forecast_series(source, task)
        changed = source.copy()
        changed.loc[changed.index[-11:-4], 'demand'] += 10_000
        after = forecast_series(changed, task)
        self.assertEqual(after['model_comparison'], before['model_comparison'])
        self.assertEqual(after['model_name'], before['model_name'])
        self.assertEqual(after['baseline_name'], before['baseline_name'])
        for first, second in zip(before['holdout']['predictions'], after['holdout']['predictions']):
            self.assertEqual(first['prediction'], second['prediction'])
            self.assertEqual(first['baseline'], second['baseline'])
        self.assertNotEqual(after['holdout']['metrics'], before['holdout']['metrics'])

    def test_row_order_does_not_change_delay_semantics_or_origin_dates(self):
        task = ForecastTask('demand', 'date', 7, 7, 2)
        self.assertEqual(forecast_series(frame(), task),
                         forecast_series(frame().sample(frac=1, random_state=12), task))


class ConfirmedAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.frame = daily_frame()
        self.semantics = confirmed(self.frame)

    def test_v3_preflight_binds_declared_delay_without_training_and_legacy_tasks_remain_v1_v2(self):
        with patch('src.forecasting._forecast') as fit:
            task = forecast_task_from_form(self.frame, self.semantics, FORM | POLICY_FORM | {'forecast_reporting_delay': '2'})
            zero = forecast_task_from_form(self.frame, self.semantics, FORM | {'forecast_reporting_delay': '0'})
        fit.assert_not_called()
        self.assertEqual(task['contract'], AVAILABILITY_TASK_CONTRACT)
        self.assertEqual(task['settings']['reporting_delay_days'], 2)
        self.assertEqual(task['availability']['reporting_delay_days'], 2)
        self.assertTrue(task['availability']['confirmed'])
        self.assertEqual(zero['contract'], AVAILABILITY_TASK_CONTRACT)
        self.assertEqual(zero['availability']['reporting_delay_days'], 0)
        self.assertEqual(forecast_task_from_form(self.frame, self.semantics, FORM)['contract'], 'confirmed-daily-forecast-task-v1')
        self.assertEqual(forecast_task_from_form(self.frame, self.semantics, FORM | POLICY_FORM)['contract'], 'confirmed-daily-forecast-task-v2')

    def test_invalid_delay_or_missing_confirmation_is_not_ignored(self):
        for value in ('', '-1', '31', '1.5', ' 2', '1e1', 'NaN', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                forecast_task_from_form(self.frame, self.semantics, FORM | {'forecast_reporting_delay': value})
        with self.assertRaises(ValueError):
            forecast_task_from_form(self.frame, self.semantics, FORM | {'forecast_reporting_delay': '2', 'forecast_available': ''})

    def test_mutating_delay_availability_contract_or_confirmation_is_rejected_before_evaluation(self):
        task = forecast_task_from_form(self.frame, self.semantics, FORM | POLICY_FORM | {'forecast_reporting_delay': '2'})
        changes = [{'availability': None}, {'availability': task['availability'] | {'reporting_delay_days': True}},
                   {'availability': task['availability'] | {'reporting_delay_days': 3}},
                   {'availability': task['availability'] | {'confirmed': False}},
                   {'availability': task['availability'] | {'confirmed': 1}},
                   {'availability': task['availability'] | {'origin_convention': 'start_of_day'}},
                   {'settings': task['settings'] | {'reporting_delay_days': 3}},
                   {'contract': 'confirmed-daily-forecast-task-v2'}, {'sha256': 'tampered'}]
        for change in changes:
            with self.subTest(change=change), patch('vibedash.forecast_tasks.forecast_series') as fit:
                with self.assertRaises(ValueError): run_confirmed_forecast(self.frame, self.semantics, task | change)
                fit.assert_not_called()
        legacy = forecast_task_from_form(self.frame, self.semantics, FORM)
        changed = copy.deepcopy(legacy)
        changed['settings']['reporting_delay_days'] = 2
        with patch('vibedash.forecast_tasks.forecast_series') as fit:
            with self.assertRaises(ValueError): run_confirmed_forecast(self.frame, self.semantics, changed)
            fit.assert_not_called()

    def test_aggregate_audit_explains_cutoffs_without_releasing_target_values_or_decision_text(self):
        task = forecast_task_from_form(self.frame, self.semantics, FORM | POLICY_FORM | {'forecast_reporting_delay': '2'})
        report = run_confirmed_forecast(self.frame, self.semantics, task)
        audit = forecast_audit_view(report)
        self.assertEqual(audit['availability']['reporting_delay_days'], 2)
        self.assertEqual(audit['future_fit_through'], '2020-06-06')
        self.assertEqual(audit['holdout_end'], '2020-06-06')
        self.assertFalse(audit['availability']['historical_vintages_verified'])
        encoded = json.dumps(audit, allow_nan=False)
        for forbidden in ('predictions', 'future_forecast', 'actual', POLICY_FORM['forecast_decision']):
            self.assertNotIn(forbidden, encoded)


if __name__ == '__main__': unittest.main()
