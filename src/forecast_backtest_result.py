"""Reconcile rolling-origin evidence with source data before web/CLI delivery.

No fitting, selection, producer scoring or producer baseline helper is called.
Consistency is not proof of fit isolation, predictive value or business validity.
"""
from collections.abc import Mapping
import math
import statistics

import pandas as pd

from src.forecasting import (
    CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE, validate_forecast_input,
)
from src.forecast_policy import validate_forecast_policy
from src.forecast_result import (
    REL_TOL, ABS_TOL, _number, _rows, _scores, _metrics, _ratio,
    _same_number, _same_ratio, _business,
)


CONTRACT = 'forecast-backtest-integrity-v1'


def _require(condition):
    if not condition:
        raise ValueError('Inconsistent historical forecast evidence.')


def _integer(value, expected):
    _require(type(value) is int and value == expected)


def _domain(domain, predictions, horizon, unit):
    if unit != 'count':
        _require(domain == {'status': 'not_assessed'})
        return False
    _require(isinstance(domain, Mapping) and domain.get('contract') == 'count-forecast-domain-v1')
    checks = domain['checks']
    scopes = ('holdout_method', 'holdout_reference', 'future')
    _require(isinstance(checks, Mapping) and set(checks) == set(scopes))
    failed = []
    for scope in scopes:
        check = checks[scope]
        _require(isinstance(check, Mapping) and set(check) == {'checked_values', 'negative_values'})
        _integer(check['checked_values'], horizon)
        negative = check['negative_values']
        _require(type(negative) is int and 0 <= negative <= horizon)
        if negative:
            failed.append(scope)
    _integer(checks['future']['negative_values'], sum(value < 0 for value in predictions))
    _require(domain['failed_scopes'] == failed)
    _require(domain['status'] == ('failed' if failed else 'nonnegative_point_forecast'))
    # Inner holdout values are not retained here; their counters can only be
    # structurally reconciled. The producer checks them against its inner run.
    return bool(failed)


def validate_historical_backtest(report, frame, task, *, decision_policy=None,
                                 target_unit='number', candidate_suite=CANDIDATE_SUITE):
    """Check every scheduled slot, including failures; never repair the report."""
    try:
        task = validate_forecast_input(frame, task)
        policy = validate_forecast_policy(decision_policy) if decision_policy is not None else None
        _require(candidate_suite in (CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE))
        _require(isinstance(report, Mapping) and report.get('contract') == 'daily-rolling-backtest-v1')
        source = frame[[task.time_column, task.target_column]].copy(deep=True)
        source[task.time_column] = pd.to_datetime(source[task.time_column])
        source = source.sort_values(task.time_column, kind='stable').reset_index(drop=True)
        dates = source[task.time_column].dt.strftime('%Y-%m-%d').tolist()
        values = [_number(value) for value in source[task.target_column]]
        if target_unit == 'count':
            _require(all(value >= 0 for value in values))
        h, delay, period = task.horizon, task.reporting_delay_days, task.seasonal_period
        end = len(source) if task.forecast_origin is not None else len(source) - delay
        minimum = max(30, 4 * period) + 4 * h + 2 * delay
        required = minimum + 3 * h + delay + (delay if task.forecast_origin is None else 0)
        enough = len(source) >= required
        objective = 'weighted_absolute_error' if policy else 'mae'
        protocol = {'windows': 3, 'horizon': h, 'seasonal_period': period,
                    'reporting_delay_days': delay, 'candidate_suite': candidate_suite,
                    'objective': objective, 'method_selection': 'inner_validation_only',
                    'window_spacing_days': h}
        _require(report['protocol'] == protocol)
        for key in ('windows', 'horizon', 'seasonal_period', 'reporting_delay_days', 'window_spacing_days'):
            _integer(report['protocol'][key], protocol[key])
        _integer(report['required_source_rows'], required)
        _integer(report['provided_source_rows'], len(source))
        _require(report['evaluation_available_through'] == dates[end - 1])
        _require(report['predictive_acceptance'] == report['production_readiness'] == 'not_established')
        windows = report['windows']
        _require(isinstance(windows, list) and len(windows) == 3)
        totals = dict(scheduled_windows=3, scored_windows=0, failed_windows=0,
                      domain_failed_windows=0, candidate_failures=0,
                      better=0, tie=0, worse=0, criteria_met_windows=0)
        checked_rows = 0
        for index, window in enumerate(windows):
            _require(isinstance(window, Mapping))
            start = end - (3 - index) * h
            available = start - delay
            _integer(window['window_index'], index)
            _integer(window['history_rows'], available)
            for key, expected in (('forecast_origin', dates[start - 1]),
                                  ('available_through', dates[available - 1]),
                                  ('start', dates[start]), ('end', dates[start + h - 1])):
                _require(window[key] == expected)
            status = window['status']
            if status in ('not_evaluated', 'error'):
                _require((not enough and status == 'not_evaluated') or (enough and status == 'error'))
                for key in ('model_name', 'baseline_name', 'metrics', 'baseline_metrics', 'comparison'):
                    _require(window[key] is None)
                fields = {'window_index', 'history_rows', 'forecast_origin', 'available_through',
                          'start', 'end', 'status', 'model_name', 'baseline_name',
                          'metrics', 'baseline_metrics', 'comparison'}
                if status == 'error':
                    _require(window.get('reason') == 'historical_forecast_evaluation_failed')
                    fields.add('reason')
                    totals['failed_windows'] += 1
                _require(set(window) == fields)
                continue
            _require(enough)
            rows = _rows(window['evidence'], dates[start:start + h], ('actual', 'prediction', 'baseline'))
            actual = values[start:start + h]
            _require(rows['actual'] == actual)
            _metrics(window['metrics'], _scores(actual, rows['prediction'], policy))
            _metrics(window['baseline_metrics'], _scores(actual, rows['baseline'], policy))
            history = values[:available]
            references = {'last_value': [history[-1]] * h,
                          'seasonal_naive': [history[-period:][step % period]
                                             for step in range(delay, delay + h)]}
            if candidate_suite == CANDIDATE_SUITE:
                references.update(historical_mean=[math.fsum(history) / len(history)] * h,
                                  historical_median=[float(statistics.median(history))] * h)
            _require(isinstance(window['references'], Mapping) and set(window['references']) == set(references))
            for name, forecast in references.items():
                recorded = window['references'][name]
                _require(isinstance(recorded, Mapping) and set(recorded) == {'metrics', 'forecast'})
                _require(isinstance(recorded['forecast'], list) and len(recorded['forecast']) == h)
                for observed, expected in zip(recorded['forecast'], forecast):
                    _same_number(observed, expected)
                _metrics(recorded['metrics'], _scores(actual, forecast, policy))
            model, baseline = window['model_name'], window['baseline_name']
            _require(model in (*references, 'lag_ridge', 'lag_random_forest') and baseline in references)
            for observed, expected in zip(rows['baseline'], references[baseline]):
                _same_number(observed, expected)
            if model in references:
                for observed, expected in zip(rows['prediction'], references[model]):
                    _same_number(observed, expected)
            selected, reference = window['metrics'][objective], window['baseline_metrics'][objective]
            outcome = ('tie' if math.isclose(selected, reference, rel_tol=REL_TOL, abs_tol=ABS_TOL)
                       else 'better' if selected < reference else 'worse')
            comparison = window['comparison']
            _require(isinstance(comparison, Mapping)
                     and set(comparison) == {'outcome', 'loss_difference', 'relative_improvement'}
                     and comparison['outcome'] == outcome)
            _same_number(comparison['loss_difference'], selected - reference)
            _same_ratio(comparison['relative_improvement'], _ratio(selected, reference))
            domain_failed = _domain(window['domain_constraints'], rows['prediction'], h, target_unit)
            failures = window['candidate_failures']
            _require(type(failures) is int and 0 <= failures <= len(references) + 1)
            _require(status == ('scored_domain_failure' if domain_failed else
                                'scored_with_candidate_failures' if failures else 'scored'))
            _business(window['business_evaluation'], policy, window['metrics'], window['baseline_metrics'],
                      domain_failed=domain_failed)
            totals['scored_windows'] += 1
            totals['domain_failed_windows'] += int(domain_failed)
            totals['candidate_failures'] += failures
            totals[outcome] += 1
            totals['criteria_met_windows'] += int(window['business_evaluation']['status'] == 'criteria_met_on_holdout')
            checked_rows += h
        _require(isinstance(report['summary'], Mapping) and set(report['summary']) == set(totals))
        for key, expected in totals.items():
            _integer(report['summary'][key], expected)
        expected_status = ('insufficient_history' if not enough else
                           'completed' if all(row['status'] == 'scored' for row in windows)
                           else 'completed_with_issues')
        _require(report['status'] == expected_status)
        return {'contract': CONTRACT, 'status': 'consistent', 'checked_windows': 3,
                'checked_scored_rows': checked_rows, 'source_facts': 'matched_for_scored_windows',
                'metrics': 'independently_recomputed', 'references': 'reconstructed_from_available_prefix',
                'relative_tolerance': REL_TOL, 'absolute_tolerance': ABS_TOL,
                'production_readiness': 'not_established'}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError, FloatingPointError) as error:
        raise ValueError('The historical forecast evidence is incomplete or inconsistent.') from error
