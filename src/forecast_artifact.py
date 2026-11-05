"""Explicit local, data-bearing JSON snapshots; never deserialize Python code.

This is not a hosted model registry, authentication boundary or live monitor.
Loading a snapshot checks its shape and accidental integrity, not authenticity.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import uuid

import numpy as np
import pandas as pd
import sklearn

from src.forecasting import (
    BASELINES, CANDIDATES, LEGACY_CANDIDATES, CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE,
    ForecastTask, _series, _lags, _fit_lag_model, validate_forecast_task,
    forecast_task_settings, validate_forecast_date, _future_window,
)
from src.forecast_domain import evaluate_count_forecast
from src.forecast_policy import validate_forecast_policy
from src.forecast_result import validate_forecast_result


CONTRACT = 'local-daily-forecast-model-v1'
PREDICTION_CONTRACT = 'local-frozen-forecast-v1'
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _history_digest(dates, values):
    return _digest([[day.strftime('%Y-%m-%d'), float(value)] for day, value in zip(dates, values)])


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('Unsupported forecast snapshot fields.')


def _finite(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('Snapshot parameters must be finite real numbers.')
    return float(value)


def _vector(value, size):
    if type(value) is not list or len(value) != size:
        raise ValueError('Invalid snapshot vector length.')
    return [_finite(item) for item in value]


def _encode_predictor(history, period, name):
    """Only objects freshly fitted here are inspected; no imported objects load."""
    if name in ('last_value', 'seasonal_naive'):
        return {'kind': name}
    if name == 'historical_mean':
        return {'kind': name, 'value': math.fsum(float(x) / len(history) for x in history)}
    if name == 'historical_median':
        ordered = sorted(float(x) for x in history)
        middle = len(ordered) // 2
        value = ordered[middle] if len(ordered) % 2 else ordered[middle - 1] / 2 + ordered[middle] / 2
        return {'kind': name, 'value': value}
    fitted = _fit_lag_model(history, period, name)
    if name == 'lag_ridge':
        scale, ridge = fitted.steps[0][1], fitted.steps[1][1]
        return {'kind': name, 'mean': scale.mean_.tolist(), 'scale': scale.scale_.tolist(),
                'coef': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_)}
    return {'kind': name, 'trees': [
        {'left': tree.tree_.children_left.tolist(), 'right': tree.tree_.children_right.tolist(),
         'feature': tree.tree_.feature.tolist(), 'threshold': tree.tree_.threshold.tolist(),
         'value': tree.tree_.value[:, 0, 0].tolist()} for tree in fitted.estimators_]}


def _validate_predictor(state, period):
    if type(state) is not dict or state.get('kind') not in CANDIDATES:
        raise ValueError('Unsupported saved predictor.')
    name = state['kind']
    features = len(_lags(period))
    if name in ('last_value', 'seasonal_naive'):
        _keys(state, ('kind',))
    elif name in ('historical_mean', 'historical_median'):
        _keys(state, ('kind', 'value'))
        _finite(state['value'])
    elif name == 'lag_ridge':
        _keys(state, ('kind', 'mean', 'scale', 'coef', 'intercept'))
        for key in ('mean', 'scale', 'coef'):
            _vector(state[key], features)
        if any(value <= 0 for value in state['scale']):
            raise ValueError('Saved scaling must be positive.')
        _finite(state['intercept'])
    else:
        _keys(state, ('kind', 'trees'))
        if type(state['trees']) is not list or len(state['trees']) != 60:
            raise ValueError('Saved forest must have exactly 60 bounded trees.')
        for tree in state['trees']:
            _keys(tree, ('left', 'right', 'feature', 'threshold', 'value'))
            if type(tree['left']) is not list or not 1 <= len(tree['left']) <= 511:
                raise ValueError('Saved tree exceeds its node bound.')
            size = len(tree['left'])
            for key in ('right', 'feature'):
                if type(tree[key]) is not list or len(tree[key]) != size:
                    raise ValueError('Saved tree arrays must match.')
            _vector(tree['threshold'], size)
            _vector(tree['value'], size)
            pending, visited = [(0, 0)], set()
            while pending:
                node, depth = pending.pop()
                if node in visited or depth > 8:
                    raise ValueError('Saved tree must be acyclic and depth-bounded.')
                visited.add(node)
                left, right, feature = (tree[key][node] for key in ('left', 'right', 'feature'))
                if any(type(value) is not int for value in (left, right, feature)):
                    raise ValueError('Saved tree indices must be integers.')
                if left == right == -1:
                    if feature != -2:
                        raise ValueError('Invalid saved leaf.')
                elif not (0 <= feature < features and node < left < size and node < right < size and left != right):
                    raise ValueError('Invalid saved tree branch.')
                else:
                    pending.extend(((left, depth + 1), (right, depth + 1)))
            if len(visited) != size:
                raise ValueError('Saved tree has unreachable nodes.')


def _predict(state, history, steps, period):
    name, lags = state['kind'], _lags(period)
    buffer = list(history)
    result = []
    cycle = list(history[-period:])
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for step in range(steps):
            if name == 'last_value':
                value = float(history[-1])
            elif name == 'seasonal_naive':
                value = float(cycle[step % period])
            elif name in ('historical_mean', 'historical_median'):
                value = state['value']
            else:
                row = np.asarray([buffer[-lag] for lag in lags], dtype=float)
                if name == 'lag_ridge':
                    normalized = (row - np.asarray(state['mean'])) / np.asarray(state['scale'])
                    value = float(normalized @ np.asarray(state['coef']) + state['intercept'])
                else:
                    # sklearn tree prediction consumes float32 features before
                    # comparing with its float64 split thresholds.
                    row = row.astype(np.float32)
                    value = 0.
                    for tree in state['trees']:
                        node = 0
                        while tree['left'][node] != -1:
                            node = (tree['left'][node] if row[tree['feature'][node]] <= tree['threshold'][node]
                                    else tree['right'][node])
                        value += tree['value'][node]
                    value /= len(state['trees'])
            if not math.isfinite(value):
                raise ValueError('Frozen prediction is non-finite.')
            result.append(float(value))
            buffer.append(float(value))
    return result


def validate_artifact(artifact):
    """Strict, bounded declarative schema. Hash is not a signature or trust seal."""
    _keys(artifact, ('contract', 'model_id', 'sha256', 'task', 'candidate_suite', 'target_unit',
                     'training', 'model', 'reference', 'decision_policy', 'evaluation', 'dependencies', 'implementation'))
    if (artifact['contract'] != CONTRACT or type(artifact['model_id']) is not str
            or not re.fullmatch('[0-9a-f]{32}', artifact['model_id'])):
        raise ValueError('Unknown local model contract or identifier.')
    if type(artifact['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', artifact['sha256']):
        raise ValueError('Invalid local model fingerprint.')
    if len(_canonical(artifact)) > MAX_ARTIFACT_BYTES:
        raise ValueError('Local model exceeds its size bound.')
    if _digest({key: value for key, value in artifact.items() if key != 'sha256'}) != artifact['sha256']:
        raise ValueError('Local model fingerprint mismatch.')
    task = validate_forecast_task(artifact['task'])
    if artifact['target_unit'] not in ('number', 'count'):
        raise ValueError('Unsupported saved target unit.')
    suite = artifact['candidate_suite']
    if suite not in (CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE):
        raise ValueError('Unknown saved candidate suite.')
    allowed = CANDIDATES if suite == CANDIDATE_SUITE else LEGACY_CANDIDATES
    for key in ('model', 'reference'):
        _validate_predictor(artifact[key], task.seasonal_period)
        if artifact[key]['kind'] not in allowed:
            raise ValueError('Saved predictor is not in the declared suite.')
    if artifact['reference']['kind'] not in BASELINES:
        raise ValueError('Saved reference must be a simple baseline.')
    if artifact['model']['kind'] in BASELINES and artifact['model'] != artifact['reference']:
        raise ValueError('A selected simple baseline must match its saved reference.')
    if artifact['decision_policy'] is not None:
        validate_forecast_policy(artifact['decision_policy'])
    training = artifact['training']
    _keys(training, ('start', 'through', 'rows', 'history_sha256', 'tail', 'issue_date'))
    minimum = max(30, 4 * task.seasonal_period) + 4 * task.horizon + 2 * task.reporting_delay_days
    if type(training['rows']) is not int or not minimum <= training['rows'] <= 5000:
        raise ValueError('Invalid saved training row count.')
    _vector(training['tail'], 2 * task.seasonal_period)
    if artifact['target_unit'] == 'count' and any(value < 0 for value in training['tail']):
        raise ValueError('Negative count history in saved model.')
    if not isinstance(training['history_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', training['history_sha256']):
        raise ValueError('Invalid saved history fingerprint.')
    for key in ('start', 'through', 'issue_date'):
        value = training[key]
        if type(value) is not str or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('Invalid saved date.')
        validate_forecast_date(value)
    if ((pd.Timestamp(training['through']) - pd.Timestamp(training['start'])).days + 1 != training['rows']
            or (pd.Timestamp(training['issue_date']) - pd.Timestamp(training['through'])).days != task.reporting_delay_days):
        raise ValueError('Saved training dates disagree with availability.')
    if task.forecast_origin is not None and task.forecast_origin != training['issue_date']:
        raise ValueError('Saved task issue date disagrees with training.')
    saved_task = ForecastTask(task.target_column, task.time_column, task.horizon,
                              task.seasonal_period, task.reporting_delay_days, training['issue_date'])
    _future_window(pd.DatetimeIndex([training['through']]), saved_task)
    _keys(artifact['evaluation'], ('holdout', 'historical_backtest_status', 'production_readiness'))
    if artifact['evaluation']['production_readiness'] != 'not_established':
        raise ValueError('A local model cannot assert operational approval.')
    evaluation = artifact['evaluation']
    if evaluation['historical_backtest_status'] not in ('not_assessed', 'insufficient_history', 'completed', 'completed_with_issues'):
        raise ValueError('Invalid historical evaluation status.')
    _keys(evaluation['holdout'], ('start', 'end', 'metrics', 'baseline_metrics'))
    expected_end = pd.Timestamp(training['through'])
    expected_start = expected_end - pd.Timedelta(days=task.horizon - 1)
    if (evaluation['holdout']['start'] != expected_start.strftime('%Y-%m-%d')
            or evaluation['holdout']['end'] != expected_end.strftime('%Y-%m-%d')):
        raise ValueError('Saved holdout dates disagree with training availability.')
    metrics = ('mae', 'rmse', 'weighted_absolute_error') if artifact['decision_policy'] else ('mae', 'rmse')
    for key in ('metrics', 'baseline_metrics'):
        _keys(evaluation['holdout'][key], metrics)
        if any(_finite(value) < 0 for value in evaluation['holdout'][key].values()):
            raise ValueError('Saved historical errors must be nonnegative.')
    _keys(artifact['implementation'], ('forecasting_sha256', 'artifact_sha256', 'forecast_policy_sha256'))
    if any(type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value)
           for value in artifact['implementation'].values()):
        raise ValueError('Invalid implementation fingerprints.')
    _keys(artifact['dependencies'], ('python', 'numpy', 'pandas', 'sklearn'))
    if any(type(value) is not str or len(value) > 64 for value in artifact['dependencies'].values()):
        raise ValueError('Invalid dependency metadata.')
    return task


def create_artifact(frame, task, report, *, decision_policy=None, target_unit='number', model_id=None):
    """Refit the already selected method once and bind it to verified evidence."""
    if model_id is not None and (not isinstance(model_id, str) or not re.fullmatch(r'[0-9a-f]{32}', model_id)):
        raise ValueError('Invalid server model identifier.')
    task = validate_forecast_task(task)
    if decision_policy is not None:
        decision_policy = validate_forecast_policy(decision_policy)
    dates, values = _series(frame, task)
    validate_forecast_result(report, frame, forecast_task_settings(task), decision_policy=decision_policy)
    if target_unit not in ('number', 'count'):
        raise ValueError('Explicit number or count unit required.')
    suite = report.get('candidate_suite')
    if suite not in (CANDIDATE_SUITE, LEGACY_CANDIDATE_SUITE):
        raise ValueError('Unknown evaluated candidate suite.')
    allowed = CANDIDATES if suite == CANDIDATE_SUITE else LEGACY_CANDIDATES
    if report['model_name'] not in allowed or report['baseline_name'] not in BASELINES or report['baseline_name'] not in allowed:
        raise ValueError('Evaluated method and reference must belong to the declared suite.')
    if target_unit == 'count' and ((values < 0).any() or evaluate_count_forecast(report, task.horizon)['status'] == 'failed'):
        raise ValueError('Cannot snapshot a domain-invalid count evaluation.')
    end = len(values) if task.forecast_origin is not None else len(values) - task.reporting_delay_days
    history, dates = values[:end], dates[:end]
    model = _encode_predictor(history, task.seasonal_period, report['model_name'])
    reference = (dict(model) if report['model_name'] == report['baseline_name'] else
                 _encode_predictor(history, task.seasonal_period, report['baseline_name']))
    future = _predict(model, history, task.horizon + task.reporting_delay_days,
                      task.seasonal_period)[task.reporting_delay_days:]
    if not np.allclose(future, [row['prediction'] for row in report['future_forecast']], rtol=1e-9, atol=1e-12):
        raise ValueError('Saved predictor does not reproduce the evaluated future refit.')
    artifact = {'contract': CONTRACT, 'model_id': model_id if model_id is not None else uuid.uuid4().hex,
                'task': forecast_task_settings(task), 'candidate_suite': report['candidate_suite'],
                'target_unit': target_unit, 'model': model, 'reference': reference,
                'training': {'start': dates[0].strftime('%Y-%m-%d'), 'through': dates[-1].strftime('%Y-%m-%d'),
                             'rows': end, 'history_sha256': _history_digest(dates, history),
                             'tail': history[-2 * task.seasonal_period:].tolist(),
                             'issue_date': report['future_forecast_origin']},
                'decision_policy': decision_policy,
                'evaluation': {'holdout': {'metrics': report['holdout']['metrics'],
                                          'baseline_metrics': report['holdout']['baseline_metrics'],
                                          'start': report['holdout']['start'], 'end': report['holdout']['end']},
                               'historical_backtest_status': report.get('historical_backtest', {}).get('status', 'not_assessed'),
                               'production_readiness': 'not_established'},
                'dependencies': {'python': platform.python_version(), 'numpy': np.__version__,
                                 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'implementation': {key: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for key, name in (('forecasting_sha256', 'forecasting.py'),
                                                     ('artifact_sha256', 'forecast_artifact.py'),
                                                     ('forecast_policy_sha256', 'forecast_policy.py'))}}
    artifact['sha256'] = _digest(artifact)
    validate_artifact(artifact)
    return json.loads(_canonical(artifact))


def predict_artifact(artifact, *, history=None, forecast_origin=None):
    """Frozen parameters; optionally append released actuals, never refit/select."""
    task = validate_artifact(artifact)
    training, delay = artifact['training'], task.reporting_delay_days
    origin = training['issue_date'] if forecast_origin is None else forecast_origin
    updated = ForecastTask(task.target_column, task.time_column, task.horizon, task.seasonal_period, delay, origin)
    if history is None:
        if origin != training['issue_date']:
            raise ValueError('A later issue date requires newly available history.')
        values, cutoff = training['tail'], training['through']
        supplied_rows = training['rows']
    else:
        dates, values = _series(history, updated)
        count = training['rows']
        if (len(values) < count or dates[0].strftime('%Y-%m-%d') != training['start']
                or _history_digest(dates[:count], values[:count]) != training['history_sha256']):
            raise ValueError('New history must preserve the complete original training prefix.')
        if artifact['target_unit'] == 'count' and (values < 0).any():
            raise ValueError('New count history contains negative observations.')
        cutoff, supplied_rows = dates[-1].strftime('%Y-%m-%d'), len(values)
    _, _, days = _future_window(pd.DatetimeIndex([cutoff]), updated)
    output = _predict(artifact['model'], values, delay + task.horizon, task.seasonal_period)[delay:]
    reference = _predict(artifact['reference'], values, delay + task.horizon, task.seasonal_period)[delay:]
    invalid = artifact['target_unit'] == 'count' and any(value < 0 for value in output + reference)
    return {'contract': PREDICTION_CONTRACT, 'status': 'domain_constraint_failed' if invalid else 'ok',
            'prediction_id': uuid.uuid4().hex, 'model_id': artifact['model_id'], 'model_sha256': artifact['sha256'],
            'forecast_origin': origin, 'available_through': cutoff, 'available_rows': supplied_rows,
            'trained_through': training['through'], 'model_name': artifact['model']['kind'],
            'baseline_name': artifact['reference']['kind'], 'parameters_refitted': False,
            'method_reselected': False, 'quality_on_new_outcomes': 'unknown',
            'production_readiness': 'not_established',
            'predictions': [{'date': day.strftime('%Y-%m-%d'), 'prediction': prediction, 'baseline': baseline}
                            for day, prediction, baseline in zip(days, output, reference)],
            'limitations': ['Frozen parameters; historical mean/median constants remain fixed at training time.',
                            'New actuals update lag context, not learned weights or method selection.',
                            'No prediction interval, measured new-outcome quality, automatic retraining or operational approval.']}


def save_artifact(artifact, path):
    """Atomic local creation, mode 0600, without replacing any existing path."""
    validate_artifact(artifact)
    data, path = _canonical(artifact), Path(path).expanduser()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.forecast-model-', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def load_artifact(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate model fields.')
            result[key] = value
        return result
    def reject_constant(_value):
        raise ValueError('Non-finite JSON is not a model.')
    with Path(path).expanduser().open('rb') as stream:
        data = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError('Local model exceeds its size bound.')
    try:
        artifact = json.loads(data, object_pairs_hook=unique, parse_constant=reject_constant)
        validate_artifact(artifact)
    except (TypeError, RecursionError, OverflowError, KeyError) as error:
        raise ValueError('Invalid local model snapshot.') from error
    return artifact
