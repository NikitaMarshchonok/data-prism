"""Explicit forecast decision criteria, fixed before temporal evaluation.

Relative error weights are preference units, not prices or financial savings.
Passing these historical criteria is never permission to deploy a model.
"""
from collections.abc import Mapping
import math

import numpy as np


CONTRACT = 'forecast-decision-policy-v1'
FIELDS = {'contract', 'decision', 'underprediction_weight', 'overprediction_weight',
          'minimum_improvement', 'maximum_mae'}


def validate_forecast_policy(value):
    if not isinstance(value, Mapping) or set(value) != FIELDS or value.get('contract') != CONTRACT:
        raise ValueError('Provide a complete supported forecast decision policy.')
    decision = value['decision']
    if not isinstance(decision, str) or not 1 <= len(decision.strip()) <= 500:
        raise ValueError('Describe the decision supported by the forecast in 1–500 characters.')
    try:
        decision.encode('utf-8')
    except UnicodeError as error:
        raise ValueError('The forecast decision must contain valid text.') from error
    result = dict(value, decision=decision.strip())
    for name in ('underprediction_weight', 'overprediction_weight', 'minimum_improvement', 'maximum_mae'):
        number = value[name]
        if name == 'maximum_mae' and number is None:
            continue
        try:
            finite = type(number) in (int, float) and math.isfinite(number)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('Forecast decision criteria must be finite numbers, not booleans or strings.')
        if name.endswith('_weight') and not 0 < number <= 1000:
            raise ValueError('Error weights must be greater than zero and at most 1000.')
        if name == 'minimum_improvement' and not 0 <= number < 1:
            raise ValueError('Required relative improvement must be at least zero and below one.')
        if name == 'maximum_mae' and number < 0:
            raise ValueError('Maximum acceptable MAE must be non-negative or omitted.')
        result[name] = float(number)
    return result


def weighted_absolute_error(actual, prediction, policy):
    """Mean(w_under * max(y - yhat, 0) + w_over * max(yhat - y, 0))."""
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if (actual.ndim != 1 or actual.shape != prediction.shape or actual.size == 0
            or not np.isfinite(actual).all() or not np.isfinite(prediction).all()):
        raise ValueError('Weighted forecast error requires matching non-empty finite one-dimensional arrays.')
    with np.errstate(over='raise', invalid='raise'):
        error = actual - prediction
        loss = float(np.mean(policy['underprediction_weight'] * np.maximum(error, 0)
                             + policy['overprediction_weight'] * np.maximum(-error, 0)))
    if not math.isfinite(loss):
        raise ValueError('The weighted forecast error is not finite.')
    return loss


def evaluate_forecast_policy(policy, metrics, baseline_metrics):
    selected, baseline = metrics['weighted_absolute_error'], baseline_metrics['weighted_absolute_error']
    improvement = 1 - selected / baseline if baseline > 0 else None
    if improvement is not None and not math.isfinite(improvement):
        improvement = None
    checks = {
        'strictly_lower_weighted_error': selected < baseline,
        'minimum_improvement': improvement is not None and improvement >= policy['minimum_improvement'],
        'maximum_mae': policy['maximum_mae'] is None or metrics['mae'] <= policy['maximum_mae'],
    }
    return {'contract': CONTRACT, 'status': ('criteria_met_on_holdout' if all(checks.values())
                                           else 'criteria_not_met_on_holdout'),
            'weighted_absolute_error': selected, 'baseline_weighted_absolute_error': baseline,
            'relative_improvement': improvement, 'checks': checks,
            'production_readiness': 'not_established', 'financial_value': 'not_assessed'}
