import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.semantic_contract import infer_semantic_contract, validate_semantic_contract
from vibedash.forecast_tasks import (
    ForecastTaskError, forecast_task_from_form, run_confirmed_forecast, forecast_audit_view,
)


def daily_frame():
    day = np.arange(160)
    return pd.DataFrame({'dteday': pd.date_range('2020-01-01', periods=160).strftime('%Y-%m-%d'),
                         'cnt': np.rint(200 + .7 * day + 30 * np.sin(day * 2 * np.pi / 7)).astype(int),
                         'casual': np.tile([20, 40], 80)})


def confirmed(frame):
    contract = infer_semantic_contract(frame, confirmed=True)
    for column in contract['columns']:
        if column['name'] == 'cnt':
            column.update(role='measure', unit='count', is_target=True, exclude_from_predictors=True)
    return validate_semantic_contract(frame, contract)


FORM = {'analysis_mode': 'forecast', 'forecast_horizon': '7', 'forecast_period': '7',
        'forecast_available': 'yes'}


class ConfirmedForecastTaskTests(unittest.TestCase):
    def setUp(self):
        self.frame = daily_frame()
        self.semantics = confirmed(self.frame)

    def test_task_is_bound_to_reviewed_target_time_units_and_availability(self):
        with patch('src.forecasting._forecast') as fit:
            task = forecast_task_from_form(self.frame, self.semantics, FORM)
        fit.assert_not_called()
        self.assertEqual(task['settings'], {'target_column': 'cnt', 'time_column': 'dteday',
                                           'horizon': 7, 'seasonal_period': 7})
        self.assertEqual(task['semantics_sha256'], self.semantics['sha256'])
        self.assertEqual(task['target_unit'], 'count')
        self.assertTrue(task['availability_confirmed'])
        self.assertRegex(task['sha256'], '^[a-f0-9]{64}$')

    def test_evidence_mode_does_not_train_forecasts_implicitly(self):
        self.assertIsNone(forecast_task_from_form(self.frame, self.semantics, {}))

    def test_invalid_or_unconfirmed_task_never_trains(self):
        for override in ({'forecast_available': ''}, {'forecast_horizon': '0'}, {'forecast_horizon': '7.0'},
                         {'forecast_period': '91'}, {'analysis_mode': 'unknown'}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                forecast_task_from_form(self.frame, self.semantics, FORM | override)
        for field in ('is_target', 'is_time'):
            changed = copy.deepcopy(self.semantics)
            for column in changed['columns']: column[field] = False
            with self.assertRaises(ValueError): forecast_task_from_form(self.frame, changed, FORM)
        changed = copy.deepcopy(self.semantics)
        changed['confirmed'] = False
        with self.assertRaises(ValueError): forecast_task_from_form(self.frame, changed, FORM)

    def test_worker_rejects_mutated_settings_binding_or_confirmation(self):
        task = forecast_task_from_form(self.frame, self.semantics, FORM)
        for mutation in ('settings', 'semantics_sha256', 'target_unit', 'sha256', 'availability_confirmed'):
            changed = copy.deepcopy(task)
            if mutation == 'settings': changed['settings']['target_column'] = 'casual'
            else: changed[mutation] = False if mutation == 'availability_confirmed' else 'tampered'
            with self.subTest(mutation=mutation), patch('vibedash.forecast_tasks.forecast_series') as fit:
                with self.assertRaises(ValueError): run_confirmed_forecast(self.frame, self.semantics, changed)
                fit.assert_not_called()

    def test_worker_rechecks_daily_history_not_only_schema(self):
        task = forecast_task_from_form(self.frame, self.semantics, FORM)
        with self.assertRaises(ValueError):
            run_confirmed_forecast(self.frame.drop(index=30), self.semantics, task)

    def test_real_evaluation_and_audit_exclude_predictive_inputs_and_raw_rows(self):
        task = forecast_task_from_form(self.frame, self.semantics, FORM)
        report = run_confirmed_forecast(self.frame, self.semantics, task)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['unused_columns'], ['casual'])
        audit = forecast_audit_view(report)
        self.assertEqual(audit['confirmed_task'], task)
        self.assertEqual(audit['holdout_metrics'], report['holdout']['metrics'])
        self.assertEqual(audit['domain_constraints'], report['domain_constraints'])
        encoded = json.dumps(audit, allow_nan=False)
        for forbidden in ('predictions', 'future_forecast', 'actual', 'casual'):
            self.assertNotIn(forbidden, encoded)

    def test_negative_count_inputs_are_rejected_and_negative_outputs_are_flagged(self):
        invalid = self.frame.copy()
        invalid.loc[20, 'cnt'] = -1
        with self.assertRaisesRegex(ValueError, 'negative observations'):
            forecast_task_from_form(invalid, self.semantics, FORM)
        task = forecast_task_from_form(self.frame, self.semantics, FORM)
        report = {'status': 'ok', 'future_forecast': [{'prediction': -1}]}
        with patch('vibedash.forecast_tasks.forecast_series', return_value=report):
            result = run_confirmed_forecast(self.frame, self.semantics, task)
        self.assertEqual(result['domain_constraints']['status'], 'failed')

    def test_worker_rejects_missing_or_inconsistent_task_before_training(self):
        for task in (None, {}, {'contract': 'unknown'}, {'contract': 'confirmed-daily-forecast-task-v1'}):
            with self.subTest(task=task), patch('vibedash.forecast_tasks.forecast_series') as fit:
                with self.assertRaises(ForecastTaskError):
                    run_confirmed_forecast(self.frame, self.semantics, task)
                fit.assert_not_called()


if __name__ == '__main__': unittest.main()
