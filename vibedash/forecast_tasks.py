"""Bind a daily forecasting task to confirmed dataset semantics."""
from dataclasses import asdict
import hashlib
import json
import re
from collections.abc import Mapping

from src.forecasting import ForecastTask, forecast_series, validate_forecast_input
from src.semantic_contract import validate_semantic_contract


CONTRACT = 'confirmed-daily-forecast-task-v1'


class ForecastTaskError(ValueError):
    pass


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _confirmed_task(frame, semantics, horizon, period, available):
    if available is not True:
        raise ForecastTaskError('Confirm that each completed day’s target is available before forecasting the next day.')
    semantics = validate_semantic_contract(frame, semantics)
    targets = [column for column in semantics['columns'] if column['is_target']]
    times = [column for column in semantics['columns'] if column['is_time']]
    if len(targets) != 1 or targets[0]['role'] != 'measure' or len(times) != 1:
        raise ForecastTaskError('Daily forecasting requires one confirmed numeric measure target and one time column.')
    try:
        task = ForecastTask(targets[0]['name'], times[0]['name'], horizon, period)
        validate_forecast_input(frame, task)
        if targets[0]['unit'] == 'count' and (frame[task.target_column] < 0).any():
            raise ValueError('The confirmed count target contains negative observations.')
    except ValueError as error:
        raise ForecastTaskError(str(error)) from error
    result = {'contract': CONTRACT, 'settings': asdict(task), 'availability_confirmed': True,
              'semantics_sha256': semantics['sha256'], 'target_unit': targets[0]['unit']}
    result['sha256'] = _fingerprint(result)
    return result


def forecast_task_from_form(frame, semantics, form):
    mode = form.get('analysis_mode', 'evidence')
    if mode == 'evidence':
        return None
    if mode != 'forecast':
        raise ForecastTaskError('Unsupported analysis mode.')
    values = []
    for name in ('forecast_horizon', 'forecast_period'):
        raw = form.get(name)
        if not isinstance(raw, str) or not re.fullmatch(r'[0-9]{1,2}', raw):
            raise ForecastTaskError('Forecast horizon and seasonal period must be explicit integers from 1 to 90 days.')
        values.append(int(raw))
    return _confirmed_task(frame, semantics, *values, form.get('forecast_available') == 'yes')


def run_confirmed_forecast(frame, semantics, stored_task):
    if not isinstance(stored_task, Mapping) or stored_task.get('contract') != CONTRACT:
        raise ForecastTaskError('The confirmed forecast task is missing or unsupported.')
    settings = stored_task.get('settings')
    if not isinstance(settings, Mapping):
        raise ForecastTaskError('The forecast settings are missing.')
    expected = _confirmed_task(frame, semantics, settings.get('horizon'), settings.get('seasonal_period'),
                               stored_task.get('availability_confirmed'))
    if dict(stored_task) != expected:
        raise ForecastTaskError('The forecast task no longer matches the confirmed target, dates or units. Review it again.')
    report = forecast_series(frame, expected['settings'])
    if report['status'] != 'ok':
        raise ForecastTaskError('The forecast evaluation could not complete; no model result was accepted.')
    report['confirmed_task'] = expected
    report['domain_constraints'] = {'status': 'not_assessed'}
    if expected['target_unit'] == 'count':
        invalid = any(row['prediction'] < 0 for row in report['future_forecast'])
        report['domain_constraints'] = {'status': 'failed' if invalid else 'nonnegative_point_forecast',
                                        'note': 'Count forecasts are expected values, not rounded integer observations.'}
    return report


def forecast_audit_view(report):
    """Aggregate-only audit metadata; never copy actuals or predictions."""
    return {'contract': report['contract'], 'confirmed_task': report['confirmed_task'],
            'status': report['predictive_support']['status'], 'model_name': report['model_name'],
            'baseline_name': report['baseline_name'], 'validation_folds': report['cv_folds'],
            'holdout_start': report['holdout']['start'], 'holdout_end': report['holdout']['end'],
            'holdout_metrics': report['holdout']['metrics'],
            'baseline_metrics': report['holdout']['baseline_metrics'],
            'mae_improvement': report['holdout']['mae_improvement'],
            'domain_constraints': report['domain_constraints'],
            'production_readiness': 'not_established'}
