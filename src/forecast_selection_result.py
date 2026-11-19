"""Check temporal validation and selection from retained daily predictions.

This validates reported evidence, not learned-model internals or fit isolation.
It never calls fitting, forecasting or the producer's scoring functions.
"""
from collections.abc import Mapping
import math
import statistics

import pandas as pd

from src.forecasting import (
    CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE, validate_forecast_input, forecast_task_settings,
)
from src.forecast_policy import validate_forecast_policy
from src.forecast_result import _rows, _metrics, _scores, _number, _same_number


CONTRACT = 'forecast-selection-integrity-v1'
# Ordered identities are part of the versioned protocol, not inferred from the
# report. Failed candidates must retain their slots in this same order.
SUITES = {
    LEGACY_CANDIDATE_SUITE: ('last_value', 'seasonal_naive', 'lag_ridge', 'lag_random_forest'),
    CANDIDATE_SUITE: ('last_value', 'seasonal_naive', 'historical_mean', 'historical_median',
                      'lag_ridge', 'lag_random_forest'),
}


def _require(condition):
    if not condition:
        raise ValueError('Inconsistent forecast validation or selection evidence.')


def _reference(history, name, horizon, delay, period):
    """Reconstruct only a reference that actually completed validation."""
    if name == 'last_value':
        return [history[-1]] * horizon
    if name == 'seasonal_naive':
        return [history[-period:][step % period] for step in range(delay, delay + horizon)]
    if name == 'historical_mean':
        return [math.fsum(value / len(history) for value in history)] * horizon
    lower, upper = statistics.median_low(history), statistics.median_high(history)
    median = lower if len(history) % 2 else lower / 2 + upper / 2
    return [median] * horizon


def validate_forecast_selection(report, frame, expected_task, *, decision_policy=None):
    """Reconcile all candidates/folds and the stable validation-only ranking."""
    try:
        task = validate_forecast_input(frame, expected_task)
        policy = validate_forecast_policy(decision_policy) if decision_policy is not None else None
        _require(isinstance(report, Mapping)
                 and report.get('status') == 'ok'
                 and report.get('validation_evidence_contract') == 'forecast-validation-evidence-v1')
        _require(report.get('task') == forecast_task_settings(task) and report.get('decision_policy') == policy)
        suite = report['candidate_suite']
        _require(suite in SUITES)
        candidates = SUITES[suite]
        baselines = candidates[:-2]
        _require(report['baseline_candidates'] == list(baselines))
        objective = 'weighted_absolute_error' if policy else 'mae'
        expected_metric = 'mean validation weighted absolute error' if policy else 'mean validation MAE'
        _require(report['selection_metric'] == expected_metric)
        source = frame[[task.time_column, task.target_column]].copy(deep=True)
        source[task.time_column] = pd.to_datetime(source[task.time_column])
        source = source.sort_values(task.time_column, kind='stable').reset_index(drop=True)
        dates = source[task.time_column].dt.strftime('%Y-%m-%d').tolist()
        values = [_number(value) for value in source[task.target_column]]
        h, period, delay = task.horizon, task.seasonal_period, task.reporting_delay_days
        end = len(source) if task.forecast_origin is not None else len(source) - delay
        selection_end = end - h - delay
        origins = [selection_end - h * remaining for remaining in (3, 2, 1)]
        folds = report['validation_folds']
        _require(isinstance(folds, list) and len(folds) == 3)
        _require(type(report['cv_folds']) is int and report['cv_folds'] == 3)
        for origin, fold in zip(origins, folds):
            available = origin - delay
            expected = {'train_start': dates[0], 'train_end': dates[available - 1],
                        'forecast_origin': dates[origin - 1], 'unavailable_days': delay,
                        'train_rows': available, 'validation_start': dates[origin],
                        'validation_end': dates[origin + h - 1], 'validation_rows': h}
            _require(fold == expected)
            _require(all(type(fold[key]) is int for key in ('unavailable_days', 'train_rows', 'validation_rows')))
        comparison = report['model_comparison']
        _require(isinstance(comparison, list) and len(comparison) == len(candidates))
        _require([row['model_name'] for row in comparison] == list(candidates))
        successful = []
        checked_rows = 0
        for candidate, row in zip(candidates, comparison):
            _require(isinstance(row, Mapping) and type(row['selected']) is bool)
            if row['status'] == 'error':
                _require(row['selected'] is False and row['mean_mae'] is None
                         and row['mean_objective_loss'] is None and row['fold_metrics'] == []
                         and row['fold_predictions'] == [])
                continue
            _require(row['status'] == 'ok')
            scores, predictions = row['fold_metrics'], row['fold_predictions']
            _require(isinstance(scores, list) and isinstance(predictions, list)
                     and len(scores) == len(predictions) == 3)
            for index, origin in enumerate(origins):
                evidence = _rows(predictions[index], dates[origin:origin + h], ('actual', 'prediction'))
                actual = values[origin:origin + h]
                _require(evidence['actual'] == actual)
                _metrics(scores[index], _scores(actual, evidence['prediction'], policy))
                if candidate in baselines:
                    reference = _reference(values[:origin - delay], candidate, h, delay, period)
                    for recorded, expected in zip(evidence['prediction'], reference):
                        _same_number(recorded, expected)
                checked_rows += h
            _require(_number(row['mean_mae']) >= 0 and _number(row['mean_objective_loss']) >= 0)
            _same_number(row['mean_mae'], math.fsum(score['mae'] for score in scores) / 3)
            _same_number(row['mean_objective_loss'], math.fsum(score[objective] for score in scores) / 3)
            successful.append(row)
        successful_baselines = [row for row in successful if row['model_name'] in baselines]
        _require(successful_baselines)
        # Preserve verified reported floats for exact-tie ordering, as the
        # producer does; independent summation may differ only at roundoff.
        model = min(successful, key=lambda row: row['mean_objective_loss'])['model_name']
        baseline = min(successful_baselines, key=lambda row: row['mean_objective_loss'])['model_name']
        _require(report['model_name'] == model and report['baseline_name'] == baseline)
        _require([row['model_name'] for row in comparison if row['selected']] == [model])
        return {'contract': CONTRACT, 'status': 'consistent', 'objective': objective,
                'checked_candidates': len(candidates), 'checked_folds': 3,
                'checked_prediction_rows': checked_rows, 'selection': 'matched_validation_ranking',
                'production_readiness': 'not_established'}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError, FloatingPointError) as error:
        raise ValueError('The forecast validation or selection evidence is incomplete or inconsistent.') from error
