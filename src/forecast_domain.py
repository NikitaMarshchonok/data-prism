"""Check returned point forecasts against an explicitly confirmed count unit.

This is a necessary non-negativity check, not domain certification. No value is
clipped, rounded, replaced, or used to reselect a method after the holdout.
"""
from collections.abc import Mapping
import math
from numbers import Real


CONTRACT = 'count-forecast-domain-v1'


def evaluate_count_forecast(report, horizon):
    """Require complete finite outputs; retain negative outputs as failed evidence.

    An incomplete/non-numeric result cannot be labelled nonnegative just because
    `any(value < 0)` was false. Such malformed results are rejected, while valid
    numerical results with negative values retain their original metrics/rows.
    Earlier validation predictions are not retained by the engine and are not
    certified by this check.
    """
    if type(horizon) is not int or not 1 <= horizon <= 90:
        raise ValueError('A supported expected forecast horizon is required.')
    if not isinstance(report, Mapping) or report.get('status') != 'ok':
        raise ValueError('A completed numerical forecast is required.')
    holdout = report.get('holdout')
    if not isinstance(holdout, Mapping):
        raise ValueError('The forecast is missing its held-out result.')
    groups = (
        ('holdout_method', holdout.get('predictions'), 'prediction'),
        ('holdout_reference', holdout.get('predictions'), 'baseline'),
        ('future', report.get('future_forecast'), 'prediction'),
    )
    checks = {}
    for scope, rows, field in groups:
        if not isinstance(rows, list) or len(rows) != horizon:
            raise ValueError(f'The {scope} output must contain the complete expected horizon.')
        negative = 0
        for row in rows:
            value = row.get(field) if isinstance(row, Mapping) else None
            try:
                finite = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f'The {scope} output must contain finite real point estimates.')
            negative += int(value < 0)
        checks[scope] = {'checked_values': len(rows), 'negative_values': negative}
    failed = [scope for scope, check in checks.items() if check['negative_values']]
    return {'contract': CONTRACT, 'status': 'failed' if failed else 'nonnegative_point_forecast',
            'checks': checks, 'failed_scopes': failed,
            'note': 'Count forecasts are point estimates, not rounded observations or guaranteed expected counts. The selected loss and method determine their interpretation. Nonnegative output is not proof of business suitability.'}
