"""Descriptive error direction and lead-time summaries after evidence checks.

These summaries never select, fit, adjust or approve forecasts. Callers must
first verify the source-bound main result and historical backtest evidence.
"""
from collections.abc import Mapping
import math
from numbers import Real

from src.forecast_policy import validate_forecast_policy


CONTRACT = 'forecast-error-diagnostics-v1'


def _number(value):
    try:
        valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('Error diagnostics require finite real observations and point estimates.')
    return float(value)


def error_profile(actual, prediction, *, decision_policy=None):
    """Signed error is prediction minus actual; contributions use all N days."""
    if not len(actual) or len(actual) != len(prediction):
        raise ValueError('Error diagnostics require matching non-empty horizons.')
    policy = validate_forecast_policy(decision_policy) if decision_policy is not None else None
    errors = [_number(_number(predicted) - _number(observed))
              for observed, predicted in zip(actual, prediction)]
    n = len(errors)
    under = [max(-error, 0.) for error in errors]
    over = [max(error, 0.) for error in errors]
    mean_under = math.fsum(value / n for value in under)
    mean_over = math.fsum(value / n for value in over)
    result = {'observations': n, 'underpredicted_days': sum(error < 0 for error in errors),
              'overpredicted_days': sum(error > 0 for error in errors),
              'exact_days': sum(error == 0 for error in errors),
              'mean_signed_error': math.fsum(error / n for error in errors),
              'underprediction_mae_contribution': mean_under,
              'overprediction_mae_contribution': mean_over,
              'mae': mean_under + mean_over,
              'maximum_underprediction': max(under), 'maximum_overprediction': max(over),
              'underprediction_weighted_contribution': (policy['underprediction_weight'] * mean_under if policy else None),
              'overprediction_weighted_contribution': (policy['overprediction_weight'] * mean_over if policy else None)}
    result['weighted_absolute_error'] = (result['underprediction_weighted_contribution']
                                        + result['overprediction_weighted_contribution'] if policy else None)
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise ValueError('Error diagnostics exceed the finite numerical range.')
    return result


def _profiles(rows, policy):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Complete numerical evidence is required for error diagnostics.')
    actual = [row['actual'] for row in rows]
    return {name: error_profile(actual, [row[field] for row in rows], decision_policy=policy)
            for name, field in (('model', 'prediction'), ('baseline', 'baseline'))}


def build_forecast_diagnostics(report):
    """Summarize verified evidence; absent/failed origins never become zeros."""
    try:
        if not isinstance(report, Mapping) or report.get('status') != 'ok':
            raise ValueError('A completed numerical report is required.')
        h = report['task']['horizon']
        if type(h) is not int or not 1 <= h <= 90:
            raise ValueError('A supported forecast horizon is required.')
        policy = report.get('decision_policy')
        policy = validate_forecast_policy(policy) if policy is not None else None
        holdout = report['holdout']['predictions']
        if not isinstance(holdout, list) or len(holdout) != h:
            raise ValueError('The final holdout must contain its complete horizon.')
        historical = report['historical_backtest']
        windows = historical['windows']
        if not isinstance(windows, list) or len(windows) != 3:
            raise ValueError('All three historical origins must remain explicit.')
        scored = []
        for window in windows:
            if window['status'] in ('error', 'not_evaluated'):
                continue
            if window['status'] not in ('scored', 'scored_domain_failure', 'scored_with_candidate_failures'):
                raise ValueError('Unsupported historical evaluation status.')
            rows = window['evidence']
            if not isinstance(rows, list) or len(rows) != h:
                raise ValueError('A scored historical origin must contain its complete horizon.')
            scored.append(rows)
        leads = []
        for lead in range(h):
            rows = [window[lead] for window in scored]
            leads.append({'lead_days': lead + 1, 'observations': len(rows),
                          **(_profiles(rows, policy) if rows else {'model': None, 'baseline': None})})
        return {'contract': CONTRACT, 'sign_convention': 'prediction_minus_actual',
                'holdout': _profiles(holdout, policy), 'historical_by_lead': leads,
                'scheduled_origins': 3, 'scored_origins': len(scored),
                'historical_status': historical['status'],
                'production_readiness': 'not_established'}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError('Forecast error diagnostics require complete, finite, verified evidence.') from error
