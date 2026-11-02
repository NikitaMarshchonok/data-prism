"""Independent delivery-boundary consistency checks, not predictive acceptance.

No private forecasting metric function is reused. Source facts, returned dates,
errors and decision flags are checked without editing or reselecting forecasts.
This does not verify fitted model internals or prove absence of training leakage.
"""
from collections.abc import Mapping
import math
from numbers import Real
import re

import pandas as pd


CONTRACT = 'forecast-result-integrity-v1'
REL_TOL = 1e-9
ABS_TOL = 1e-12


def _number(value):
    try:
        valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('Forecast evidence requires finite real numbers.')
    return float(value)


def _same_number(observed, expected):
    if expected is None:
        if observed is not None:
            raise ValueError('An undefined forecast ratio must not become a numerical gain.')
        return
    observed = _number(observed)
    # Zero/undefined reference errors have distinct semantics, even at tiny scales.
    if ((observed == 0 or expected == 0) and observed != expected
            or not math.isclose(observed, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)):
        raise ValueError('Reported forecast metrics disagree with the returned evidence.')


def _scores(actual, predicted, policy):
    errors = [value - prediction for value, prediction in zip(actual, predicted)]
    result = {'mae': math.fsum(abs(error) for error in errors) / len(errors),
              'rmse': math.sqrt(math.fsum(error * error for error in errors) / len(errors))}
    if policy is not None:
        result['weighted_absolute_error'] = math.fsum(
            policy['underprediction_weight'] * max(error, 0.)
            + policy['overprediction_weight'] * max(-error, 0.) for error in errors) / len(errors)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Independently recomputed forecast errors must be finite.')
    return result


def _ratio(selected, baseline):
    value = 1 - selected / baseline if baseline > 0 else None
    return value if value is None or math.isfinite(value) else None


def _metrics(observed, expected):
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        raise ValueError('Forecast error fields do not match the confirmed scoring policy.')
    for key, value in expected.items():
        if _number(observed[key]) < 0:
            raise ValueError('Forecast errors cannot be negative.')
        _same_number(observed[key], value)


def _same_ratio(observed, expected):
    if observed is not None and _number(observed) > 1:
        raise ValueError('Error reduction cannot exceed one.')
    _same_number(observed, expected)


def _rows(value, dates, fields):
    if not isinstance(value, list) or len(value) != len(dates):
        raise ValueError('Forecast evidence must contain the full expected horizon.')
    result = {field: [] for field in fields}
    for row, date in zip(value, dates):
        if not isinstance(row, Mapping) or row.get('date') != date:
            raise ValueError('Forecast dates do not match the confirmed evaluation window.')
        for field in fields:
            result[field].append(_number(row.get(field)))
    return result


def validate_forecast_result(report, frame, expected_task, *, decision_policy=None):
    """Check a completed result against prevalidated input/settings before delivery.

    The caller must already have validated daily input, task and optional policy.
    Numerical tolerance handles summation roundoff, not business acceptance.
    Decision predicates are checked on the verified reported metrics so rounding
    differences in an independent sum do not silently move a policy threshold.
    """
    try:
        if not isinstance(report, Mapping) or report.get('status') != 'ok':
            raise ValueError('A completed forecast result is required.')
        if report.get('task') != expected_task or report.get('decision_policy') != decision_policy:
            raise ValueError('Forecast evidence does not match the confirmed task and decision policy.')
        target, time = expected_task['target_column'], expected_task['time_column']
        h, delay = expected_task['horizon'], expected_task.get('reporting_delay_days', 0)
        if type(h) is not int or not 1 <= h <= 90 or type(delay) is not int or not 0 <= delay <= 30:
            raise ValueError('Validated forecast settings are required.')
        source = frame[[time, target]].copy()
        source[time] = pd.to_datetime(source[time], errors='raise')
        source = source.sort_values(time, kind='stable')
        dates = source[time].dt.strftime('%Y-%m-%d').tolist()
        origin = expected_task.get('forecast_origin', dates[-1])
        if not isinstance(origin, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', origin):
            raise ValueError('A validated daily issue date is required.')
        end = len(source) if 'forecast_origin' in expected_task else len(source) - delay
        start = end - h
        if start - delay < 1:
            raise ValueError('Validated forecast history is required.')
        explicit = 'forecast_origin' in expected_task
        availability = report['availability']
        expected_availability = {
            'contract': 'forecast-origin-availability-v2' if explicit else 'forecast-origin-availability-v1',
            'history_mode': 'available_history_only' if explicit else 'retrospective_complete_series',
            'origin_source': 'explicit_task_date' if explicit else 'last_input_date',
            'latest_input_date': dates[-1], 'reporting_delay_days': delay,
            'origin_convention': 'end_of_completed_day',
            'evaluation_outcomes_available_through': dates[end - 1],
            'selection_outcomes_available_through': dates[start - delay - 1],
            'unavailable_tail_rows': 0 if explicit else delay,
            'source': 'caller_assertion', 'historical_vintages_verified': False,
        }
        if (not isinstance(availability, Mapping) or dict(availability) != expected_availability
                or type(availability.get('reporting_delay_days')) is not int
                or type(availability.get('unavailable_tail_rows')) is not int
                or availability.get('historical_vintages_verified') is not False):
            raise ValueError('Forecast availability metadata does not match the confirmed convention.')
        expected_selection = 'mean validation weighted absolute error' if decision_policy else 'mean validation MAE'
        if report.get('selection_metric') != expected_selection:
            raise ValueError('The reported selection objective does not match the confirmed scoring policy.')
        holdout_dates = dates[start:end]
        expected_facts = [_number(value) for value in source[target].iloc[start:end]]
        future_dates = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=h).strftime('%Y-%m-%d').tolist()
        holdout = report['holdout']
        if (holdout.get('start') != holdout_dates[0] or holdout.get('end') != holdout_dates[-1]
                or holdout.get('train_end') != dates[start - delay - 1]
                or holdout.get('forecast_origin') != dates[start - 1]
                or report.get('future_forecast_origin') != origin
                or report.get('future_fit_through') != dates[end - 1]
                or type(report.get('test_rows')) is not int or report['test_rows'] != h):
            raise ValueError('Forecast windows do not match the confirmed input and availability convention.')
        rows = _rows(holdout['predictions'], holdout_dates, ('actual', 'prediction', 'baseline'))
        if rows['actual'] != expected_facts:  # Source facts are not an approximate metric.
            raise ValueError('Held-out facts do not match the uploaded target observations.')
        _rows(report['future_forecast'], future_dates, ('prediction',))
        _metrics(holdout['metrics'], _scores(expected_facts, rows['prediction'], decision_policy))
        _metrics(holdout['baseline_metrics'], _scores(expected_facts, rows['baseline'], decision_policy))
        method, baseline = holdout['metrics'], holdout['baseline_metrics']
        _same_ratio(holdout['mae_improvement'], _ratio(method['mae'], baseline['mae']))

        if any(not isinstance(report.get(field), str) or not report[field]
               for field in ('model_name', 'baseline_name')):
            raise ValueError('Forecast method and reference identities must be explicit.')
        same_method = report['model_name'] == report['baseline_name']
        if same_method and rows['prediction'] != rows['baseline']:
            raise ValueError('The same selected method and reference cannot have different holdout forecasts.')
        support = ('baseline_selected' if same_method else 'observed_holdout_improvement'
                   if method['mae'] < baseline['mae'] else 'no_holdout_improvement')
        if report.get('predictive_support') != {'status': support, 'statistical_significance': 'not_assessed',
                                                'production_readiness': 'not_established'}:
            raise ValueError('Forecast support claims do not match the reported historical comparison.')
        business = report['business_evaluation']
        if decision_policy is None:
            if business != {'status': 'not_specified', 'production_readiness': 'not_established'}:
                raise ValueError('Business suitability cannot be asserted without a confirmed decision policy.')
        else:
            selected_loss, reference_loss = method['weighted_absolute_error'], baseline['weighted_absolute_error']
            improvement = _ratio(selected_loss, reference_loss)
            checks = {
                'strictly_lower_weighted_error': selected_loss < reference_loss,
                'minimum_improvement': improvement is not None and improvement >= decision_policy['minimum_improvement'],
                'maximum_mae': decision_policy['maximum_mae'] is None or method['mae'] <= decision_policy['maximum_mae'],
            }
            status = 'criteria_met_on_holdout' if all(checks.values()) else 'criteria_not_met_on_holdout'
            business_fields = {'contract', 'status', 'weighted_absolute_error', 'baseline_weighted_absolute_error',
                               'relative_improvement', 'checks', 'production_readiness', 'financial_value'}
            if (not isinstance(business, Mapping) or set(business) != business_fields
                    or business.get('contract') != decision_policy['contract']
                    or business.get('status') != status or business.get('checks') != checks
                    or any(type(value) is not bool for value in business.get('checks', {}).values())
                    or business.get('production_readiness') != 'not_established'
                    or business.get('financial_value') != 'not_assessed'):
                raise ValueError('Business-criteria claims disagree with the confirmed policy and verified metrics.')
            for field, expected in (('weighted_absolute_error', selected_loss),
                                    ('baseline_weighted_absolute_error', reference_loss)):
                _same_number(business.get(field), expected)
            _same_ratio(business.get('relative_improvement'), improvement)
        return {'contract': CONTRACT, 'status': 'consistent', 'checked_holdout_rows': h, 'checked_future_rows': h,
                'source_facts': 'matched', 'metrics': 'independently_recomputed',
                'relative_tolerance': REL_TOL, 'absolute_tolerance': ABS_TOL,
                'production_readiness': 'not_established'}
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError, FloatingPointError) as error:
        # Closed diagnostics only; never include a source value or parser dump.
        raise ValueError('The forecast result is incomplete or inconsistent.') from error
