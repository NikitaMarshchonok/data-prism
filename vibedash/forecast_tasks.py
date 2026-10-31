"""Bind a daily forecasting task to confirmed dataset semantics."""
import hashlib
import json
import re
from collections.abc import Mapping

from src.forecasting import (
    ForecastTask, forecast_series, validate_forecast_input, forecast_task_settings,
    AVAILABILITY_CONTRACT, ORIGIN_AVAILABILITY_CONTRACT,
)
from src.semantic_contract import validate_semantic_contract
from src.forecast_policy import validate_forecast_policy, CONTRACT as DECISION_POLICY_CONTRACT


CONTRACT = 'confirmed-daily-forecast-task-v1'
POLICY_TASK_CONTRACT = 'confirmed-daily-forecast-task-v2'
AVAILABILITY_TASK_CONTRACT = 'confirmed-daily-forecast-task-v3'
ORIGIN_TASK_CONTRACT = 'confirmed-daily-forecast-task-v4'
POLICY_FIELDS = ('forecast_decision', 'forecast_under_weight', 'forecast_over_weight',
                 'forecast_minimum_improvement', 'forecast_maximum_mae', 'forecast_policy_confirmed')


class ForecastTaskError(ValueError):
    pass


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _confirmed_task(frame, semantics, horizon, period, available, policy=None, policy_confirmed=False,
                    *, reporting_delay_days=0, availability_declared=False, forecast_origin=None):
    if available is not True:
        raise ForecastTaskError('Review and confirm when completed-day targets become available at each forecast origin.')
    if forecast_origin is not None and not availability_declared:
        raise ForecastTaskError('An explicit forecast origin requires an explicitly reviewed reporting delay, including zero.')
    semantics = validate_semantic_contract(frame, semantics)
    targets = [column for column in semantics['columns'] if column['is_target']]
    times = [column for column in semantics['columns'] if column['is_time']]
    if len(targets) != 1 or targets[0]['role'] != 'measure' or len(times) != 1:
        raise ForecastTaskError('Daily forecasting requires one confirmed numeric measure target and one time column.')
    try:
        task = ForecastTask(targets[0]['name'], times[0]['name'], horizon, period, reporting_delay_days, forecast_origin)
        validate_forecast_input(frame, task)
        if targets[0]['unit'] == 'count' and (frame[task.target_column] < 0).any():
            raise ValueError('The confirmed count target contains negative observations.')
    except ValueError as error:
        raise ForecastTaskError(str(error)) from error
    result = {'contract': CONTRACT, 'settings': forecast_task_settings(task), 'availability_confirmed': True,
              'semantics_sha256': semantics['sha256'], 'target_unit': targets[0]['unit']}
    if policy is not None:
        if policy_confirmed is not True:
            raise ForecastTaskError('Confirm the decision criteria before starting the forecast evaluation.')
        result.update(contract=POLICY_TASK_CONTRACT, decision_policy=validate_forecast_policy(policy),
                      decision_policy_confirmed=True)
    if availability_declared:
        result.update(contract=AVAILABILITY_TASK_CONTRACT, availability={
            'contract': AVAILABILITY_CONTRACT, 'reporting_delay_days': reporting_delay_days,
            'origin_convention': 'end_of_completed_day', 'confirmed': True,
        })
        if forecast_origin is not None:
            result['contract'] = ORIGIN_TASK_CONTRACT
            result['availability'].update(contract=ORIGIN_AVAILABILITY_CONTRACT,
                                          forecast_origin=forecast_origin, history_mode='available_history_only')
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
    declared = 'forecast_reporting_delay' in form
    delay = 0
    if declared:
        raw = form.get('forecast_reporting_delay')
        if not isinstance(raw, str) or not re.fullmatch(r'[0-9]{1,2}', raw) or not 0 <= int(raw) <= 30:
            raise ForecastTaskError('Target reporting delay must be an explicit integer from 0 to 30 whole days.')
        delay = int(raw)
    raw_origin = form.get('forecast_origin', '')
    if not isinstance(raw_origin, str):
        raise ForecastTaskError('Forecast origin must be an ISO date or empty for retrospective mode.')
    origin = raw_origin if raw_origin != '' else None
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
                           policy, form.get('forecast_policy_confirmed') == 'yes',
                           reporting_delay_days=delay, availability_declared=declared, forecast_origin=origin)


def run_confirmed_forecast(frame, semantics, stored_task):
    if not isinstance(stored_task, Mapping) or stored_task.get('contract') not in {CONTRACT, POLICY_TASK_CONTRACT, AVAILABILITY_TASK_CONTRACT, ORIGIN_TASK_CONTRACT}:
        raise ForecastTaskError('The confirmed forecast task is missing or unsupported.')
    settings = stored_task.get('settings')
    if not isinstance(settings, Mapping):
        raise ForecastTaskError('The forecast settings are missing.')
    if stored_task['contract'] == POLICY_TASK_CONTRACT and stored_task.get('decision_policy') is None:
        raise ForecastTaskError('The confirmed forecast decision policy is missing.')
    explicit_origin = stored_task['contract'] == ORIGIN_TASK_CONTRACT
    declared = stored_task['contract'] in {AVAILABILITY_TASK_CONTRACT, ORIGIN_TASK_CONTRACT}
    delay = 0
    origin = None
    if declared:
        availability = stored_task.get('availability')
        availability_contract = ORIGIN_AVAILABILITY_CONTRACT if explicit_origin else AVAILABILITY_CONTRACT
        if not isinstance(availability, Mapping) or availability.get('contract') != availability_contract:
            raise ForecastTaskError('The confirmed forecast availability contract is missing or unsupported.')
        if availability.get('confirmed') is not True:
            raise ForecastTaskError('The target reporting delay must be explicitly confirmed.')
        delay = availability.get('reporting_delay_days')
        if explicit_origin:
            origin = availability.get('forecast_origin')
            if not isinstance(origin, str) or not origin:
                raise ForecastTaskError('The confirmed forecast origin is missing.')
    expected = _confirmed_task(frame, semantics, settings.get('horizon'), settings.get('seasonal_period'),
                               stored_task.get('availability_confirmed'), stored_task.get('decision_policy'),
                               stored_task.get('decision_policy_confirmed', False),
                               reporting_delay_days=delay, availability_declared=declared, forecast_origin=origin)
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
            'availability': report['availability'],
            'holdout_origin': report['holdout']['forecast_origin'],
            'holdout_fit_through': report['holdout']['train_end'],
            'future_origin': report['future_forecast_origin'],
            'future_fit_through': report['future_fit_through'],
            'production_readiness': 'not_established'}
