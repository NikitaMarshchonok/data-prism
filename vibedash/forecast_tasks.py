"""Bind a daily forecasting task to confirmed dataset semantics."""
from dataclasses import asdict
import hashlib
import json
import re
from collections.abc import Mapping

from src.forecasting import ForecastTask, forecast_series, validate_forecast_input
from src.semantic_contract import validate_semantic_contract
from src.forecast_policy import validate_forecast_policy, CONTRACT as DECISION_POLICY_CONTRACT


CONTRACT = 'confirmed-daily-forecast-task-v1'
POLICY_TASK_CONTRACT = 'confirmed-daily-forecast-task-v2'
POLICY_FIELDS = ('forecast_decision', 'forecast_under_weight', 'forecast_over_weight',
                 'forecast_minimum_improvement', 'forecast_maximum_mae', 'forecast_policy_confirmed')


class ForecastTaskError(ValueError):
    pass


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _confirmed_task(frame, semantics, horizon, period, available, policy=None, policy_confirmed=False):
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
    if policy is not None:
        if policy_confirmed is not True:
            raise ForecastTaskError('Confirm the decision criteria before starting the forecast evaluation.')
        result.update(contract=POLICY_TASK_CONTRACT, decision_policy=validate_forecast_policy(policy),
                      decision_policy_confirmed=True)
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
    policy = None
    if any(name in form for name in POLICY_FIELDS):
        try:
            policy = validate_forecast_policy({
                'contract': DECISION_POLICY_CONTRACT, 'decision': form.get('forecast_decision'),
                'underprediction_weight': float(form.get('forecast_under_weight', '')),
                'overprediction_weight': float(form.get('forecast_over_weight', '')),
                'minimum_improvement': float(form.get('forecast_minimum_improvement', '')) / 100,
                'maximum_mae': (None if form.get('forecast_maximum_mae') == '' else
                                float(form.get('forecast_maximum_mae', ''))),
            })
        except (TypeError, ValueError, OverflowError) as error:
            raise ForecastTaskError('Provide valid, complete forecast decision criteria: decision, positive error weights, improvement below 100%, and an optional non-negative MAE limit.') from error
    return _confirmed_task(frame, semantics, *values, form.get('forecast_available') == 'yes',
                           policy, form.get('forecast_policy_confirmed') == 'yes')


def run_confirmed_forecast(frame, semantics, stored_task):
    if not isinstance(stored_task, Mapping) or stored_task.get('contract') not in {CONTRACT, POLICY_TASK_CONTRACT}:
        raise ForecastTaskError('The confirmed forecast task is missing or unsupported.')
    settings = stored_task.get('settings')
    if not isinstance(settings, Mapping):
        raise ForecastTaskError('The forecast settings are missing.')
    if stored_task['contract'] == POLICY_TASK_CONTRACT and stored_task.get('decision_policy') is None:
        raise ForecastTaskError('The confirmed forecast decision policy is missing.')
    expected = _confirmed_task(frame, semantics, settings.get('horizon'), settings.get('seasonal_period'),
                               stored_task.get('availability_confirmed'), stored_task.get('decision_policy'),
                               stored_task.get('decision_policy_confirmed', False))
    if dict(stored_task) != expected:
        raise ForecastTaskError('The forecast task no longer matches the confirmed target, dates or units. Review it again.')
    report = forecast_series(frame, expected['settings'], decision_policy=expected.get('decision_policy'))
    if report['status'] != 'ok':
        raise ForecastTaskError('The forecast evaluation could not complete; no model result was accepted.')
    report['confirmed_task'] = expected
    report['domain_constraints'] = {'status': 'not_assessed'}
    if expected['target_unit'] == 'count':
        invalid = any(row['prediction'] < 0 for row in report['future_forecast'])
        report['domain_constraints'] = {'status': 'failed' if invalid else 'nonnegative_point_forecast',
                                        'note': 'Count forecasts are expected values, not rounded integer observations.'}
        if invalid and report.get('business_evaluation', {}).get('status') == 'criteria_met_on_holdout':
            report['business_evaluation']['status'] = 'domain_constraint_failed'
    return report


def forecast_audit_view(report):
    """Aggregate-only audit metadata; never copy actuals or predictions."""
    confirmed_metadata = dict(report['confirmed_task'])
    if confirmed_metadata.get('decision_policy') is not None:
        # Decision text may contain business context. Keep its bound fingerprint,
        # numeric criteria and closed statuses, never the user's free text.
        confirmed_metadata['decision_policy'] = {
            key: value for key, value in confirmed_metadata['decision_policy'].items() if key != 'decision'
        }
    return {'contract': report['contract'], 'confirmed_task': confirmed_metadata,
            'status': report['predictive_support']['status'], 'model_name': report['model_name'],
            'baseline_name': report['baseline_name'], 'validation_folds': report['cv_folds'],
            'holdout_start': report['holdout']['start'], 'holdout_end': report['holdout']['end'],
            'holdout_metrics': report['holdout']['metrics'],
            'baseline_metrics': report['holdout']['baseline_metrics'],
            'mae_improvement': report['holdout']['mae_improvement'],
            'domain_constraints': report['domain_constraints'],
            'selection_metric': report['selection_metric'],
            'business_evaluation': report['business_evaluation'],
            'production_readiness': 'not_established'}
