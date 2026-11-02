"""Scalar forecast scoring and simple references, independent of fitted models."""
import math
from numbers import Real
import statistics

import numpy as np

from src.forecasting import CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE


def score_predictions(actual, prediction, *, decision_policy=None):
    if len(actual) != len(prediction) or not len(actual):
        raise ValueError('Actuals and predictions must have the same non-empty length.')
    values = list(actual) + list(prediction)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not math.isfinite(float(value)) for value in values):
        raise ValueError('Scores require finite real values.')
    errors = [float(observed) - float(predicted) for observed, predicted in zip(actual, prediction)]
    result = {'mae': math.fsum(abs(error) for error in errors) / len(errors),
              'rmse': math.sqrt(math.fsum(error * error for error in errors) / len(errors))}
    if decision_policy is not None:
        result['weighted_absolute_error'] = math.fsum(
            decision_policy['underprediction_weight'] * max(error, 0.)
            + decision_policy['overprediction_weight'] * max(-error, 0.) for error in errors) / len(errors)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Scores must be finite.')
    return result


def baseline_predictions(history, horizon, delay, *, period=7, candidate_suite=LEGACY_CANDIDATE_SUITE):
    """Predict after issue date, advancing the observed cycle through the gap."""
    if len(history) < period:
        raise ValueError('At least one complete seasonal cycle is required for all references.')
    cycle = list(history[-period:])
    result = {'last_value': [float(history[-1])] * horizon,
              'seasonal_naive': [float(cycle[step % period]) for step in range(delay, delay + horizon)]}
    if candidate_suite == CANDIDATE_SUITE:
        result['historical_mean'] = [math.fsum(history) / len(history)] * horizon
        result['historical_median'] = [float(statistics.median(history))] * horizon
    elif candidate_suite != LEGACY_CANDIDATE_SUITE:
        raise ValueError('Unknown baseline suite.')
    return result
