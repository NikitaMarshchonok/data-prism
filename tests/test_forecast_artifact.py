"""Local model lifecycle: round trip, frozen fits, release cutoffs and safe JSON."""
from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluate_forecast
import predict_forecast
from src.forecast_artifact import (
    CONTRACT, MAX_ARTIFACT_BYTES, _digest, _encode_predictor, _predict, _validate_predictor,
    create_artifact, load_artifact, predict_artifact, save_artifact, validate_artifact,
)
from src.forecasting import CANDIDATES, ForecastTask, _forecast, forecast_series
from tests.test_forecasting import frame
from tests.test_forecast_policy import POLICY


TASK = ForecastTask('demand', 'date', 7, 7, 2, '2020-06-10')


def resign(artifact):
    artifact['sha256'] = _digest({key: value for key, value in artifact.items() if key != 'sha256'})
    return artifact


class ForecastArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = frame()
        cls.report = forecast_series(cls.data, TASK)
        cls.artifact = create_artifact(cls.data, TASK, cls.report)

    def test_every_predictor_json_roundtrip_matches_fresh_engine_for_deduplicated_and_weekly_lags(self):
        rng = np.random.default_rng(83)
        for period in (1, 3, 7):
            history = np.linspace(50, 180, 160) + rng.normal(0, 5, 160)
            for name in CANDIDATES:
                with self.subTest(period=period, method=name):
                    state = json.loads(json.dumps(_encode_predictor(history, period, name), allow_nan=False))
                    _validate_predictor(state, period)
                    expected = _forecast(history, 98, period, name)
                    with patch('src.forecast_artifact._fit_lag_model', side_effect=AssertionError('must not fit')):
                        actual = _predict(state, history[-2 * period:], 98, period)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_forest_uses_float32_branch_inputs_like_fitted_estimator(self):
        tree = {'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [0, -2, -2],
                'threshold': [1.00000003, -2., -2.], 'value': [15., 10., 20.]}
        state = {'kind': 'lag_random_forest', 'trees': [tree] * 60}
        _validate_predictor(state, 1)
        self.assertEqual(_predict(state, [0., 1.00000004], 1, 1), [10.])

    def test_saved_model_reproduces_evaluated_future_and_survives_process_restart(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'private.forecast-model.json'
            save_artifact(self.artifact, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            loaded = load_artifact(path)
            self.assertEqual(loaded, self.artifact)
            with patch('src.forecast_artifact._fit_lag_model', side_effect=AssertionError('must not fit')):
                prediction = predict_artifact(loaded)
            completed = subprocess.run([sys.executable, 'predict_forecast.py', '--model', str(path)],
                                       capture_output=True, text=True, timeout=30, check=True)
        other_process = json.loads(completed.stdout)
        self.assertEqual(prediction['predictions'], other_process['predictions'])
        self.assertEqual(prediction['model_sha256'], other_process['model_sha256'])
        np.testing.assert_allclose([row['prediction'] for row in prediction['predictions']],
                                   [row['prediction'] for row in self.report['future_forecast']], rtol=1e-12)
        self.assertFalse(prediction['parameters_refitted'])
        self.assertFalse(prediction['method_reselected'])
        self.assertEqual(prediction['quality_on_new_outcomes'], 'unknown')

    def test_new_released_actuals_update_lags_but_cannot_fit_or_reselect(self):
        extended = pd.concat([self.data, pd.DataFrame({'date': pd.date_range('2020-06-09', periods=3),
                                                     'demand': [450., 420., 460.]})], ignore_index=True)
        before = json.dumps(self.artifact, sort_keys=True)
        with (patch('src.forecast_artifact._fit_lag_model', side_effect=AssertionError('must not fit')),
              patch('src.forecasting.forecast_series', side_effect=AssertionError('must not select'))):
            result = predict_artifact(self.artifact, history=extended, forecast_origin='2020-06-13')
        self.assertEqual(json.dumps(self.artifact, sort_keys=True), before)
        self.assertEqual(result['available_through'], '2020-06-11')
        self.assertEqual(result['trained_through'], '2020-06-08')
        self.assertEqual(result['predictions'][0]['date'], '2020-06-14')
        expected = _predict(self.artifact['model'], extended.demand.tolist(), 9, 7)[2:]
        self.assertEqual([row['prediction'] for row in result['predictions']], expected)
        original = predict_artifact(self.artifact)
        self.assertNotEqual(original['predictions'][0]['prediction'], result['predictions'][0]['prediction'])

    def test_changed_training_prefix_stale_or_unreleased_history_and_implicit_future_are_rejected(self):
        changed = self.data.copy()
        changed.loc[0, 'demand'] += 1.
        for history, origin in ((changed, '2020-06-10'), (self.data, '2020-06-11'),
                                (self.data, '2020-06-09'), (self.data.iloc[1:], '2020-06-10')):
            with self.subTest(origin=origin, rows=len(history)), self.assertRaises(ValueError):
                predict_artifact(self.artifact, history=history, forecast_origin=origin)
        with self.assertRaises(ValueError):
            predict_artifact(self.artifact, forecast_origin='2020-06-11')

    def test_retrospective_snapshot_excludes_unreleased_tail(self):
        task = ForecastTask('demand', 'date', 7, 7, 2)
        before = create_artifact(self.data, task, forecast_series(self.data, task))
        changed = self.data.copy()
        changed.loc[158:, 'demand'] += 10000
        after = create_artifact(changed, task, forecast_series(changed, task))
        for key in ('model', 'reference', 'training', 'evaluation'):
            self.assertEqual(before[key], after[key])
        self.assertEqual(before['training']['through'], '2020-06-06')
        self.assertEqual(before['training']['issue_date'], '2020-06-08')

    def test_snapshot_rejects_inconsistent_evidence_and_mislabelled_fitted_method(self):
        for mutate in ('metric', 'future', 'task', 'method'):
            result = copy.deepcopy(self.report)
            if mutate == 'metric': result['holdout']['metrics']['mae'] = -1.
            elif mutate == 'future': result['future_forecast'][0]['prediction'] += 10.
            elif mutate == 'task': result['task']['horizon'] = 6
            else: result['model_name'] = 'lag_random_forest'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                create_artifact(self.data, TASK, result)

    def test_policy_is_bound_snapshot_is_detached_and_count_invalid_results_cannot_save(self):
        policy = copy.deepcopy(POLICY)
        report = forecast_series(self.data, TASK, decision_policy=policy)
        saved = create_artifact(self.data, TASK, report, decision_policy=policy, target_unit='count')
        self.assertEqual(saved['decision_policy'], policy)
        self.assertEqual(saved['target_unit'], 'count')
        report['holdout']['metrics']['mae'] = 9999
        policy['decision'] = 'Changed after snapshot'
        validate_artifact(saved)
        with self.assertRaises(ValueError): create_artifact(self.data, TASK, self.report, target_unit='currency')
        data = self.data.copy()
        data.loc[0, 'demand'] = -1
        with self.assertRaises(ValueError):
            create_artifact(data, TASK, forecast_series(data, TASK), target_unit='count')

    def test_constant_mean_and_median_parameters_do_not_silently_update_on_new_actuals(self):
        for name in ('historical_mean', 'historical_median'):
            state = _encode_predictor(self.data.demand.to_numpy(), 7, name)
            before = _predict(state, self.data.demand.tolist(), 9, 7)
            self.assertEqual(_predict(state, self.data.demand.tolist() + [10000.], 9, 7), before)

    def test_digest_and_schema_reject_tampering_even_when_attacker_rehashes_invalid_state(self):
        corrupt = copy.deepcopy(self.artifact)
        corrupt['model']['intercept'] += 1.
        with self.assertRaises(ValueError): validate_artifact(corrupt)
        mutations = [lambda x: x.update(contract='unknown'), lambda x: x.update(untrusted='field'),
                     lambda x: x.update(model_id=11111111111111111111111111111111),
                     lambda x: x['model'].update(scale=[0., 1., 1.]),
                     lambda x: x['model'].update(coef=[True, 1., 1.]),
                     lambda x: x['training'].update(tail=[]),
                     lambda x: x['training'].update(issue_date='2020-06-09'),
                     lambda x: x['training'].update(rows=True),
                     lambda x: x['evaluation'].update(production_readiness='approved'),
                     lambda x: x['evaluation']['holdout']['metrics'].update(mae=-1.),
                     lambda x: x['evaluation'].update(holdout={'untrusted': 'field'})]
        for mutate in mutations:
            corrupt = copy.deepcopy(self.artifact)
            mutate(corrupt)
            with self.assertRaises(ValueError): validate_artifact(resign(corrupt))

    def test_forest_cycles_unreachable_nodes_bad_features_and_excess_depth_are_rejected(self):
        state = _encode_predictor(self.data.demand.to_numpy(), 7, 'lag_random_forest')
        for key, value in (('left', 0), ('right', 9999), ('feature', 9)):
            corrupt = copy.deepcopy(state)
            corrupt['trees'][0][key][0] = value
            with self.assertRaises(ValueError): _validate_predictor(corrupt, 7)
        corrupt = copy.deepcopy(state)
        corrupt['trees'].pop()
        with self.assertRaises(ValueError): _validate_predictor(corrupt, 7)

    def test_unknown_evaluated_suite_fails_before_snapshot_fit(self):
        report = copy.deepcopy(self.report)
        report['candidate_suite'] = 'unknown'
        with patch('src.forecast_artifact._fit_lag_model') as fit, self.assertRaises(ValueError):
            create_artifact(self.data, TASK, report)
        fit.assert_not_called()

    def test_loaded_baseline_must_match_reference_and_no_extra_values_are_trusted(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['model'] = {'kind': 'historical_mean', 'value': 10.}
        artifact['reference'] = {'kind': 'historical_mean', 'value': 11.}
        with self.assertRaises(ValueError): validate_artifact(resign(artifact))
        artifact['reference']['value'] = 10.
        validate_artifact(resign(artifact))

    def test_loaded_snapshot_and_frozen_issue_date_obey_the_same_date_bounds(self):
        for origin in ('2262-04-10', '9999-01-01'):
            with (self.subTest(origin=origin),
                  patch('src.forecast_artifact._predict') as predict,
                  self.assertRaises(ValueError)):
                predict_artifact(self.artifact, forecast_origin=origin)
            predict.assert_not_called()
        artifact = copy.deepcopy(self.artifact)
        artifact['task']['forecast_origin'] = '2262-04-10'
        artifact['training'].update(start='2261-10-31', through='2262-04-08', issue_date='2262-04-10')
        # Keep all metadata internally consistent except the overflowing future.
        start = pd.Timestamp('2262-04-08') - pd.Timedelta(days=artifact['training']['rows'] - 1)
        artifact['training']['start'] = start.strftime('%Y-%m-%d')
        artifact['evaluation']['holdout'].update(start='2262-04-02', end='2262-04-08')
        with patch('src.forecast_artifact._predict') as predict, self.assertRaises(ValueError):
            validate_artifact(resign(artifact))
        predict.assert_not_called()

    def test_existing_destination_and_symlinks_are_not_overwritten_and_no_temp_remains(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'existing.json'
            path.write_text('user data')
            with self.assertRaises(FileExistsError): save_artifact(self.artifact, path)
            self.assertEqual(path.read_text(), 'user data')
            link = Path(directory) / 'link.json'
            link.symlink_to(path)
            with self.assertRaises(FileExistsError): save_artifact(self.artifact, link)
            self.assertEqual(path.read_text(), 'user data')
            self.assertEqual(sorted(item.name for item in Path(directory).iterdir()), ['existing.json', 'link.json'])

    def test_load_rejects_pickle_duplicate_keys_nonfinite_oversized_and_nested_payloads(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for raw in (b'\x80\x04not a JSON model', b'{"contract":1,"contract":2}', b'{"value":NaN}',
                        b'[' * 2000 + b'0' + b']' * 2000, b' ' * (MAX_ARTIFACT_BYTES + 1)):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): load_artifact(path)

    def test_failed_write_does_not_leave_partial_snapshot_or_temporary_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            with patch('src.forecast_artifact.os.link', side_effect=OSError('controlled failure')):
                with self.assertRaises(OSError): save_artifact(self.artifact, path)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_shuffled_history_preserves_binding_and_new_columns_are_not_predictors(self):
        ordered = predict_artifact(self.artifact, history=self.data, forecast_origin='2020-06-10')
        shuffled = self.data.sample(frac=1, random_state=42).copy()
        shuffled['private_unused_column'] = 1e20
        changed = predict_artifact(self.artifact, history=shuffled, forecast_origin='2020-06-10')
        self.assertEqual(ordered['predictions'], changed['predictions'])
        self.assertNotIn('private_unused_column', json.dumps(changed))

    def test_new_count_prediction_failure_remains_visible_without_clipping(self):
        # Explicit adversarial codec fixture, not a saved production model.
        saved = copy.deepcopy(self.artifact)
        saved['target_unit'] = 'count'
        saved['model'] = {'kind': 'lag_ridge', 'mean': [0., 0., 0.], 'scale': [1., 1., 1.],
                          'coef': [0., 0., 0.], 'intercept': -1.}
        result = predict_artifact(resign(saved))
        self.assertEqual(result['status'], 'domain_constraint_failed')
        self.assertTrue(all(row['prediction'] == -1. for row in result['predictions']))
        self.assertEqual(result['quality_on_new_outcomes'], 'unknown')

    def test_cli_explicit_save_and_prediction_no_overwrite_or_retraining(self):
        def invoke(function, args):
            stream = StringIO()
            with redirect_stdout(stream): code = function(args)
            return code, json.loads(stream.getvalue())
        with TemporaryDirectory() as directory:
            source, saved = Path(directory) / 'history.csv', Path(directory) / 'private.forecast-model.json'
            self.data.to_csv(source, index=False)
            args = ['--csv', str(source), '--target', 'demand', '--time', 'date', '--horizon', '7',
                    '--reporting-delay-days', '2', '--forecast-origin', '2020-06-10', '--save-model', str(saved)]
            code, result = invoke(evaluate_forecast.main, args)
            self.assertEqual(code, 0)
            self.assertEqual(result['model_artifact']['status'], 'saved')
            original = saved.read_bytes()
            code, result = invoke(evaluate_forecast.main, args)
            self.assertEqual(code, 1)
            self.assertEqual(result['model_artifact']['status'], 'error')
            self.assertEqual(saved.read_bytes(), original)
            code, result = invoke(predict_forecast.main, ['--model', str(saved)])
            self.assertEqual(code, 0)
            self.assertFalse(result['parameters_refitted'])
            code, result = invoke(predict_forecast.main, ['--model', str(saved), '--history-csv', str(source)])
            self.assertEqual(code, 1)
            self.assertNotIn(directory, json.dumps(result))


if __name__ == '__main__':
    unittest.main()
