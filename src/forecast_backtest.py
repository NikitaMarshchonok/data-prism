"""Three fixed past issue dates for one uploaded daily series.

Each window simulates the existing service procedure on a private available
prefix, then scores its future forecast against separately held outer facts.
Outer losses never reselect the final future model or change its policy.
"""
from collections import Counter
import math

import pandas as pd

from src.forecasting import (
    CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE, ForecastTask, forecast_series,
    forecast_task_settings, validate_forecast_input,
)
from src.forecast_policy import validate_forecast_policy, evaluate_forecast_policy
from src.forecast_domain import evaluate_count_forecast
from src.forecast_result import validate_forecast_result
from src.forecast_scoring import score_predictions, baseline_predictions


CONTRACT = 'daily-rolling-backtest-v1'
WINDOWS = 3


def _comparison(selected, baseline):
    delta = selected - baseline
    relative = 1 - selected / baseline if baseline > 0 else None
    return {'outcome': 'tie' if math.isclose(selected, baseline, rel_tol=1e-9, abs_tol=1e-12)
            else 'better' if delta < 0 else 'worse',
            'loss_difference': delta,
            'relative_improvement': relative if relative is None or math.isfinite(relative) else None}


def historical_forecast_backtest(frame, task, *, decision_policy=None, target_unit='number',
                                 candidate_suite=CANDIDATE_SUITE):
    """Fixed bounded protocol, not automatic acceptance or model monitoring."""
    task = validate_forecast_input(frame, task)
    policy = validate_forecast_policy(decision_policy) if decision_policy is not None else None
    if candidate_suite not in (CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE):
        raise ValueError('Unsupported backtest candidate suite.')
    source = frame[[task.time_column, task.target_column]].copy(deep=True)
    source[task.time_column] = pd.to_datetime(source[task.time_column])
    source = source.sort_values(task.time_column, kind='stable').reset_index(drop=True)
    if target_unit == 'count' and (source[task.target_column] < 0).any():
        raise ValueError('The confirmed count target contains negative observations.')
    h, delay = task.horizon, task.reporting_delay_days
    end = len(source) if task.forecast_origin is not None else len(source) - delay
    dates = source[task.time_column].dt.strftime('%Y-%m-%d').tolist()
    minimum = max(30, 4 * task.seasonal_period) + 4 * h + 2 * delay
    objective = 'weighted_absolute_error' if policy else 'mae'
    windows = []
    for index in range(WINDOWS):
        start = end - (WINDOWS - index) * h
        available_rows = start - delay
        windows.append({'window_index': index, 'forecast_origin': dates[start - 1],
                        'available_through': dates[available_rows - 1],
                        'start': dates[start], 'end': dates[start + h - 1],
                        'history_rows': available_rows, 'status': 'not_evaluated',
                        'model_name': None, 'baseline_name': None, 'metrics': None,
                        'baseline_metrics': None, 'comparison': None})
    required_rows = minimum + WINDOWS * h + delay + (delay if task.forecast_origin is None else 0)
    enough_history = all(row['history_rows'] >= minimum for row in windows)
    if enough_history:
        for window in windows:
            start = end - (WINDOWS - window['window_index']) * h
            # No gap or outer horizon values can enter this private training input.
            prefix = source.iloc[:window['history_rows']].copy(deep=True)
            origin_task = ForecastTask(task.target_column, task.time_column, h, task.seasonal_period,
                                       delay, window['forecast_origin'])
            actual = source[task.target_column].iloc[start:start + h].astype(float).tolist()
            try:
                report = forecast_series(prefix, origin_task, decision_policy=policy, candidate_suite=candidate_suite)
                validate_forecast_result(report, prefix, forecast_task_settings(origin_task), decision_policy=policy)
                if report['candidate_suite'] != candidate_suite:
                    raise ValueError('Unexpected candidate suite.')
                forecasts = [row['prediction'] for row in report['future_forecast']]
                references = baseline_predictions(prefix[task.target_column].astype(float).tolist(), h, delay,
                                                  period=task.seasonal_period, candidate_suite=candidate_suite)
                name = report['baseline_name']
                if name not in references:
                    raise ValueError('The selected reference is not part of this protocol.')
                scores = score_predictions(actual, forecasts, decision_policy=policy)
                reference_scores = {name: score_predictions(actual, values, decision_policy=policy)
                                    for name, values in references.items()}
                domain = evaluate_count_forecast(report, h) if target_unit == 'count' else {'status': 'not_assessed'}
                business = (evaluate_forecast_policy(policy, scores, reference_scores[name]) if policy else
                            {'status': 'not_specified', 'production_readiness': 'not_established'})
                if domain['status'] == 'failed' and business['status'] == 'criteria_met_on_holdout':
                    business['status'] = 'domain_constraint_failed'
                failures = sum(row['status'] != 'ok' for row in report['model_comparison'])
                status = ('scored_domain_failure' if domain['status'] == 'failed' else
                          'scored_with_candidate_failures' if failures else 'scored')
                window.update(status=status, model_name=report['model_name'], baseline_name=name,
                              metrics=scores, baseline_metrics=reference_scores[name],
                              comparison=_comparison(scores[objective], reference_scores[name][objective]),
                              candidate_failures=failures, domain_constraints=domain,
                              business_evaluation=business,
                              references={name: {'metrics': reference_scores[name], 'forecast': values}
                                          for name, values in references.items()},
                              evidence=[{'date': dates[start + step], 'actual': actual[step],
                                         'prediction': forecasts[step], 'baseline': references[name][step]}
                                        for step in range(h)])
            except Exception:
                # Retain each scheduled failure without publishing data-bearing
                # exception text or retrying with a favourable replacement.
                window.update(status='error', reason='historical_forecast_evaluation_failed')
    scored = [row for row in windows if row['metrics'] is not None]
    counts = Counter(row['comparison']['outcome'] for row in scored)
    summary = {'scheduled_windows': WINDOWS, 'scored_windows': len(scored),
               'failed_windows': sum(row['status'] == 'error' for row in windows),
               'domain_failed_windows': sum(row['status'] == 'scored_domain_failure' for row in windows),
               'candidate_failures': sum(row.get('candidate_failures', 0) for row in windows),
               'better': counts['better'], 'tie': counts['tie'], 'worse': counts['worse'],
               'criteria_met_windows': sum(row.get('business_evaluation', {}).get('status') == 'criteria_met_on_holdout' for row in windows)}
    return {'contract': CONTRACT, 'status': ('insufficient_history' if not enough_history else
                                           'completed' if all(row['status'] == 'scored' for row in windows)
                                           else 'completed_with_issues'),
            'protocol': {'windows': WINDOWS, 'horizon': h, 'seasonal_period': task.seasonal_period,
                         'reporting_delay_days': delay, 'candidate_suite': candidate_suite,
                         'objective': objective, 'method_selection': 'inner_validation_only',
                         'window_spacing_days': h},
            'evaluation_available_through': dates[end - 1], 'required_source_rows': required_rows,
            'provided_source_rows': len(source), 'summary': summary, 'windows': windows,
            'predictive_acceptance': 'not_established', 'production_readiness': 'not_established',
            'limitations': [
                'Three dependent historical origins for this one series, not new clients or an unseen certification set.',
                'Every origin reruns the same candidate selection on its available prefix; outer facts never enter that run.',
                'Later origins may use earlier released outcomes; no outer loss reselects the final future method or changes criteria.',
                'Fixed release delays are asserted, not verified historical vintages. No confidence interval or live monitoring is provided.',
                'All three origins require sufficient history; the protocol is not shortened to hide missing or adverse windows.',
            ]}


def backtest_audit_view(report):
    """Aggregate-only view: no actuals, forecasts, source rows or decision text."""
    window_fields = ('window_index', 'forecast_origin', 'available_through', 'start', 'end', 'history_rows',
                     'status', 'model_name', 'baseline_name', 'metrics', 'baseline_metrics', 'comparison',
                     'candidate_failures', 'domain_constraints', 'business_evaluation', 'reason')
    return {key: report[key] for key in ('contract', 'status', 'protocol', 'evaluation_available_through',
                                        'required_source_rows', 'provided_source_rows', 'summary',
                                        'predictive_acceptance', 'production_readiness')} | {
        'windows': [{key: row[key] for key in window_fields if key in row} for row in report['windows']]}
