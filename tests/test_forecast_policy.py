"""Independent arithmetic and selection oracles for predeclared decision criteria."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.forecast_policy import (
    CONTRACT, validate_forecast_policy, weighted_absolute_error, evaluate_forecast_policy,
)
from src.forecasting import forecast_series
from tests.test_forecasting import frame, TASK
from tests.test_forecast_tasks import FORM, daily_frame, confirmed
from vibedash.forecast_tasks import forecast_task_from_form, run_confirmed_forecast, forecast_audit_view


POLICY = {'contract': CONTRACT, 'decision': 'Plan next week staffing',
          'underprediction_weight': 5., 'overprediction_weight': 1.,
          'minimum_improvement': .1, 'maximum_mae': None}
POLICY_FORM = {'forecast_decision': POLICY['decision'], 'forecast_under_weight': '5',
               'forecast_over_weight': '1', 'forecast_minimum_improvement': '10',
               'forecast_maximum_mae': '', 'forecast_policy_confirmed': 'yes'}


class ForecastPolicyTests(unittest.TestCase):
    def test_asymmetric_loss_matches_hand_calculated_oracle(self):
        # Under by 2: penalty 10; over by 3: penalty 3; mean = 6.5.
        self.assertEqual(weighted_absolute_error([10, 10], [8, 13], POLICY), 6.5)
        reverse = POLICY | {'underprediction_weight': 1., 'overprediction_weight': 5.}
        self.assertEqual(weighted_absolute_error([10, 10], [8, 13], reverse), 8.5)
        equal = POLICY | {'underprediction_weight': 1.}
        self.assertEqual(weighted_absolute_error([10, 10], [8, 13], equal), 2.5)
        self.assertEqual(weighted_absolute_error([10, 10], [10, 10], POLICY), 0)

    def test_invalid_policies_fail_closed_before_training(self):
        invalid = [None, {}, POLICY | {'unknown': True}, POLICY | {'decision': ''},
                   POLICY | {'decision': 'x' * 501}, POLICY | {'decision': '\ud800'},
                   POLICY | {'contract': 'unknown'}]
        for name in ('underprediction_weight', 'overprediction_weight', 'minimum_improvement', 'maximum_mae'):
            for value in (True, '1', float('nan'), float('inf'), 10 ** 500, -1):
                invalid.append(POLICY | {name: value})
        invalid += [POLICY | {'underprediction_weight': 0}, POLICY | {'overprediction_weight': 1001},
                    POLICY | {'minimum_improvement': 1}]
        for policy in invalid:
            with self.subTest(policy=repr(policy)):
                with self.assertRaises(ValueError): validate_forecast_policy(policy)
                if policy is not None:  # None intentionally means legacy MAE, no business policy.
                    with patch('src.forecasting._forecast') as fit:
                        report = forecast_series(frame(), TASK, decision_policy=policy)
                    self.assertEqual(report['status'], 'error')
                    fit.assert_not_called()

    def test_objective_changes_model_and_baseline_using_validation_only(self):
        values = {'last_value': 9., 'seasonal_naive': 12., 'lag_ridge': 11.5, 'lag_random_forest': 8.}
        def predict(history, horizon, period, candidate):
            return np.repeat(values[candidate], horizon)
        data = frame(np.repeat(10., 160))
        changed = data.copy()
        changed.loc[changed.index[-7:], 'demand'] = 100.
        with patch('src.forecasting._forecast', side_effect=predict):
            default = forecast_series(data, TASK)
            weighted = forecast_series(data, TASK, decision_policy=POLICY)
            mutated = forecast_series(changed, TASK, decision_policy=POLICY)
        self.assertEqual(default['model_name'], 'last_value')
        self.assertEqual(weighted['model_name'], 'lag_ridge')
        self.assertEqual(weighted['baseline_name'], 'seasonal_naive')
        self.assertEqual(weighted['model_comparison'], mutated['model_comparison'])
        self.assertEqual(weighted['model_name'], mutated['model_name'])
        self.assertEqual(weighted['business_evaluation']['weighted_absolute_error'], 1.5)
        self.assertEqual(weighted['business_evaluation']['relative_improvement'], .25)
        self.assertEqual(weighted['business_evaluation']['status'], 'criteria_met_on_holdout')
        self.assertEqual(mutated['business_evaluation']['status'], 'criteria_not_met_on_holdout')
        self.assertEqual(weighted['business_evaluation']['production_readiness'], 'not_established')
        self.assertEqual(weighted['business_evaluation']['financial_value'], 'not_assessed')

    def test_thresholds_are_conjunctive_and_zero_baseline_does_not_invent_a_gain(self):
        measured = {'mae': 5., 'weighted_absolute_error': 5.}
        baseline = {'mae': 10., 'weighted_absolute_error': 10.}
        self.assertEqual(evaluate_forecast_policy(POLICY, measured, baseline)['status'], 'criteria_met_on_holdout')
        for policy in (POLICY | {'minimum_improvement': .6}, POLICY | {'maximum_mae': 4.}):
            self.assertEqual(evaluate_forecast_policy(policy, measured, baseline)['status'], 'criteria_not_met_on_holdout')
        zero = {'mae': 0., 'weighted_absolute_error': 0.}
        result = evaluate_forecast_policy(POLICY | {'minimum_improvement': 0}, zero, zero)
        self.assertIsNone(result['relative_improvement'])
        self.assertFalse(result['checks']['strictly_lower_weighted_error'])
        self.assertEqual(result['status'], 'criteria_not_met_on_holdout')
        extreme = evaluate_forecast_policy(POLICY, {'mae': 1., 'weighted_absolute_error': 1e308},
                                            {'mae': 1., 'weighted_absolute_error': 1e-308})
        self.assertIsNone(extreme['relative_improvement'])
        json.dumps(extreme, allow_nan=False)

    def test_web_policy_is_confirmed_bound_and_omits_decision_text_from_audit(self):
        data = daily_frame()
        semantics = confirmed(data)
        task = forecast_task_from_form(data, semantics, FORM | POLICY_FORM)
        self.assertEqual(task['contract'], 'confirmed-daily-forecast-task-v2')
        self.assertEqual(task['decision_policy'], POLICY)
        self.assertTrue(task['decision_policy_confirmed'])
        report = run_confirmed_forecast(data, semantics, task)
        audit = forecast_audit_view(report)
        encoded = json.dumps(audit, allow_nan=False)
        self.assertNotIn(POLICY['decision'], encoded)
        self.assertNotIn('predictions', encoded)
        self.assertEqual(audit['confirmed_task']['sha256'], task['sha256'])
        self.assertEqual(audit['business_evaluation'], report['business_evaluation'])
        self.assertEqual(audit['confirmed_task']['decision_policy']['underprediction_weight'], 5.)
        for mutation in ('decision', 'underprediction_weight', 'maximum_mae'):
            changed = copy.deepcopy(task)
            changed['decision_policy'][mutation] = 'Other decision' if mutation == 'decision' else 3.
            with self.subTest(mutation=mutation), patch('vibedash.forecast_tasks.forecast_series') as fit:
                with self.assertRaises(ValueError): run_confirmed_forecast(data, semantics, changed)
                fit.assert_not_called()

    def test_missing_partial_or_unconfirmed_web_criteria_are_rejected(self):
        data = daily_frame()
        semantics = confirmed(data)
        for fields in ({'forecast_decision': 'Staffing'}, POLICY_FORM | {'forecast_policy_confirmed': ''},
                       POLICY_FORM | {'forecast_under_weight': 'nan'},
                       POLICY_FORM | {'forecast_minimum_improvement': '100'}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                forecast_task_from_form(data, semantics, FORM | fields)
        legacy = forecast_task_from_form(data, semantics, FORM)
        self.assertEqual(legacy['contract'], 'confirmed-daily-forecast-task-v1')

    def test_count_domain_failure_cannot_retain_business_criteria_success(self):
        data = daily_frame()
        semantics = confirmed(data)
        task = forecast_task_from_form(data, semantics, FORM | POLICY_FORM)
        fake = {'status': 'ok', 'future_forecast': [{'prediction': -1}],
                'business_evaluation': {'status': 'criteria_met_on_holdout'}}
        with patch('vibedash.forecast_tasks.forecast_series', return_value=fake):
            result = run_confirmed_forecast(data, semantics, task)
        self.assertEqual(result['business_evaluation']['status'], 'domain_constraint_failed')


if __name__ == '__main__':
    unittest.main()
