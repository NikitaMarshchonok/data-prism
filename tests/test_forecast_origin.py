"""Issue-date forecasting accepts released history, never invented recent targets."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecasting import ForecastTask, _forecast, forecast_series, validate_forecast_input
from tests.test_forecasting import frame
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from tests.test_forecast_policy import POLICY_FORM
from vibedash.forecast_tasks import forecast_task_from_form, run_confirmed_forecast, forecast_audit_view


class ForecastOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = ForecastTask('demand', 'date', 7, 7, 2, '2020-06-10')
        cls.report = forecast_series(frame(), cls.task)

    def test_available_history_requires_no_unknown_tail_and_forecasts_after_explicit_origin(self):
        report = self.report
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['contract'], 'daily-forecast-v3')
        self.assertEqual(report['availability']['contract'], 'forecast-origin-availability-v2')
        self.assertEqual(report['availability']['history_mode'], 'available_history_only')
        self.assertEqual(report['availability']['latest_input_date'], '2020-06-08')
        self.assertEqual(report['availability']['unavailable_tail_rows'], 0)
        self.assertEqual(report['future_forecast_origin'], '2020-06-10')
        self.assertEqual(report['future_fit_through'], '2020-06-08')
        self.assertEqual(report['future_forecast'][0]['date'], '2020-06-11')
        self.assertEqual(report['future_forecast'][-1]['date'], '2020-06-17')
        self.assertEqual(report['holdout']['end'], '2020-06-08')
        json.dumps(report, allow_nan=False)

    def test_explicit_and_equivalent_retrospective_origins_have_identical_model_results(self):
        retrospective = forecast_series(frame(size=162), ForecastTask('demand', 'date', 7, 7, 2))
        for key in ('model_name', 'baseline_name', 'model_comparison', 'validation_folds',
                    'holdout', 'train_rows', 'test_rows', 'future_fit_through',
                    'future_forecast_origin', 'future_forecast', 'predictive_support'):
            self.assertEqual(self.report[key], retrospective[key], key)

    def test_omitting_origin_retains_prior_retrospective_meaning(self):
        report = forecast_series(frame(), ForecastTask('demand', 'date', 7, 7, 2))
        self.assertEqual(report['contract'], 'daily-forecast-v2')
        self.assertEqual(report['future_forecast_origin'], '2020-06-08')
        self.assertEqual(report['future_fit_through'], '2020-06-06')
        self.assertNotIn('forecast_origin', report['task'])

    def test_zero_delay_with_explicit_origin_preserves_numerical_zero_delay_results(self):
        before = forecast_series(frame(), ForecastTask('demand', 'date', 7))
        after = forecast_series(frame(), ForecastTask('demand', 'date', 7, forecast_origin='2020-06-08'))
        self.assertEqual(after['contract'], 'daily-forecast-v3')
        for key in ('model_comparison', 'holdout', 'future_forecast'):
            self.assertEqual(before[key], after[key])

    def test_invalid_or_ambiguous_origin_never_trains(self):
        fields = {'target_column': 'demand', 'time_column': 'date', 'horizon': 7}
        for origin in ('', '2020-2-01', '10/06/2020', '2020-02-30', '2020-06-08T00:00:00',
                       '2020-06-08Z', ' 2020-06-08', '9999-01-01', True, 20200608, [], {}):
            with self.subTest(origin=origin), patch('src.forecasting._forecast') as fit:
                report = forecast_series(frame(), fields | {'forecast_origin': origin})
                self.assertEqual(report['status'], 'error')
                self.assertEqual(report['future_forecast'], [])
                fit.assert_not_called()

    def test_future_dates_stale_history_missing_days_and_placeholder_targets_fail_before_fit(self):
        future = frame(size=161)
        stale = frame(size=159)
        missing = frame().drop(index=30)
        placeholder = frame(size=162)
        placeholder.loc[placeholder.index[-2:], 'demand'] = np.nan
        for data in (future, stale, missing, placeholder):
            with self.subTest(rows=len(data)), patch('src.forecasting._forecast') as fit:
                report = forecast_series(data, self.task)
                self.assertEqual(report['status'], 'error')
                fit.assert_not_called()
                with self.assertRaises(ValueError): validate_forecast_input(data, self.task)
        self.assertIn('after origin', forecast_series(future, self.task)['message'])
        self.assertIn('stale', forecast_series(stale, self.task)['message'])

    def test_forecast_window_overflow_is_rejected_during_preflight_not_after_training(self):
        data = frame().assign(date=pd.date_range(end='2262-04-10', periods=160))
        task = ForecastTask('demand', 'date', 7, forecast_origin='2262-04-10')
        with patch('src.forecasting._forecast') as fit:
            self.assertEqual(forecast_series(data, task)['status'], 'error')
            with self.assertRaises(ValueError): validate_forecast_input(data, task)
            fit.assert_not_called()

    def test_minimum_available_history_reserves_selection_gap_and_training_embargo(self):
        size, delay, horizon = 30 + 4 * 3 + 2 * 10, 10, 3
        data = frame(size=size)
        origin = (data.date.iloc[-1] + pd.Timedelta(days=delay)).strftime('%Y-%m-%d')
        task = ForecastTask('demand', 'date', horizon, 7, delay, origin)
        with patch('src.forecasting.CANDIDATES', ('last_value', 'seasonal_naive')):
            report = forecast_series(data, task)
            # Keep the same cutoff when removing an early observation.
            short = forecast_series(data.iloc[1:], task)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['validation_folds'][0]['train_rows'], 30)
        self.assertEqual(short['status'], 'error')

    def test_real_fit_prefixes_gap_lengths_and_final_refit_use_available_history_only(self):
        values, calls = frame().demand.to_numpy(), []
        def observe(history, horizon, period, candidate):
            calls.append((history.copy(), horizon))
            return _forecast(history, horizon, period, candidate)
        with patch('src.forecasting._forecast', side_effect=observe):
            report = forecast_series(frame(), self.task)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual([len(history) for history, _ in calls[:12]], [128, 135, 142] * 4)
        self.assertEqual(len(calls[-1][0]), 160)
        for history, horizon in calls:
            np.testing.assert_array_equal(history, values[:len(history)])
            self.assertEqual(horizon, 9)
        for fold in report['validation_folds']:
            self.assertLessEqual(fold['validation_end'], report['holdout']['train_end'])

    def test_holdout_and_pre_holdout_release_gap_cannot_leak_into_selection(self):
        data = frame()
        holdout_changed = data.copy()
        holdout_changed.loc[holdout_changed.index[-7:], 'demand'] += 10000
        after = forecast_series(holdout_changed, self.task)
        self.assertEqual(after['model_comparison'], self.report['model_comparison'])
        for first, second in zip(self.report['holdout']['predictions'], after['holdout']['predictions']):
            self.assertEqual(first['prediction'], second['prediction'])
        gap_changed = data.copy()
        gap_changed.loc[gap_changed.index[-9:-7], 'demand'] += 10000
        after = forecast_series(gap_changed, self.task)
        self.assertEqual(after['model_comparison'], self.report['model_comparison'])
        self.assertEqual(after['holdout'], self.report['holdout'])

    def test_naive_seasonal_bridge_oracle_advances_cycle_to_day_after_issue_date(self):
        data = frame(np.tile([11., 22., 33., 44., 55., 66., 77.], 24))
        delay, horizon = 10, 3
        origin = (data.date.iloc[-1] + pd.Timedelta(days=delay)).strftime('%Y-%m-%d')
        with patch('src.forecasting.CANDIDATES', ('seasonal_naive',)):
            report = forecast_series(data, ForecastTask('demand', 'date', horizon, 7, delay, origin))
        expected = [float(data.demand.iloc[-7 + (step % 7)]) for step in range(delay, delay + horizon)]
        self.assertEqual([row['prediction'] for row in report['future_forecast']], expected)
        self.assertEqual(len(report['future_forecast']), horizon)

    def test_row_order_and_unused_proxies_do_not_change_origin_or_predictions(self):
        data = frame().assign(future_proxy=np.arange(160) + 1_000_000).sample(frac=1, random_state=7)
        report = forecast_series(data, self.task)
        self.assertEqual(report['unused_columns'], ['future_proxy'])
        for key in ('model_comparison', 'holdout', 'availability', 'future_forecast'):
            self.assertEqual(report[key], self.report[key])


class ConfirmedOriginTests(unittest.TestCase):
    def setUp(self):
        self.frame = daily_frame()
        self.semantics = confirmed(self.frame)
        self.form = FORM | POLICY_FORM | {'forecast_reporting_delay': '2', 'forecast_origin': '2020-06-10'}

    def test_v4_binds_issue_date_and_available_history_and_preflight_does_not_train(self):
        with patch('src.forecasting._forecast') as fit:
            task = forecast_task_from_form(self.frame, self.semantics, self.form)
        fit.assert_not_called()
        self.assertEqual(task['contract'], 'confirmed-daily-forecast-task-v4')
        self.assertEqual(task['settings']['forecast_origin'], '2020-06-10')
        self.assertEqual(task['availability']['contract'], 'forecast-origin-availability-v2')
        self.assertEqual(task['availability']['forecast_origin'], '2020-06-10')
        report = run_confirmed_forecast(self.frame, self.semantics, task)
        audit = forecast_audit_view(report)
        self.assertEqual(audit['future_origin'], '2020-06-10')
        self.assertEqual(audit['future_fit_through'], '2020-06-08')
        self.assertNotIn('predictions', json.dumps(audit))
        self.assertNotIn(POLICY_FORM['forecast_decision'], json.dumps(audit))

    def test_origin_requires_declared_delay_and_empty_browser_field_keeps_v3(self):
        without_delay = self.form.copy()
        without_delay.pop('forecast_reporting_delay')
        with self.assertRaises(ValueError): forecast_task_from_form(self.frame, self.semantics, without_delay)
        for origin in (None, ' ', '2020-06-09', '2020-06-11'):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                forecast_task_from_form(self.frame, self.semantics, self.form | {'forecast_origin': origin})
        task = forecast_task_from_form(self.frame, self.semantics, self.form | {'forecast_origin': ''})
        self.assertEqual(task['contract'], 'confirmed-daily-forecast-task-v3')
        self.assertNotIn('forecast_origin', task['settings'])

    def test_mutated_dates_history_mode_or_version_cannot_train(self):
        task = forecast_task_from_form(self.frame, self.semantics, self.form)
        changes = [{'settings': task['settings'] | {'forecast_origin': '2020-06-08'}},
                   {'availability': task['availability'] | {'forecast_origin': '2020-06-11'}},
                   {'availability': task['availability'] | {'forecast_origin': None}},
                   {'availability': task['availability'] | {'history_mode': 'retrospective_complete_series'}},
                   {'contract': 'confirmed-daily-forecast-task-v3'}, {'sha256': 'tampered'}]
        for change in changes:
            with self.subTest(change=change), patch('vibedash.forecast_tasks.forecast_series') as fit:
                with self.assertRaises(ValueError): run_confirmed_forecast(self.frame, self.semantics, task | change)
                fit.assert_not_called()
        legacy = forecast_task_from_form(self.frame, self.semantics, self.form | {'forecast_origin': ''})
        changed = copy.deepcopy(legacy)
        changed['settings']['forecast_origin'] = '2020-06-10'
        with patch('vibedash.forecast_tasks.forecast_series') as fit:
            with self.assertRaises(ValueError): run_confirmed_forecast(self.frame, self.semantics, changed)
            fit.assert_not_called()


if __name__ == '__main__': unittest.main()
