"""Explicitly retained web forecasts and matched daily outcomes."""
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import io
import json
import re

from src.forecast_scoring import score_predictions
from src.forecasting import validate_forecast_date


RETENTION_DAYS = 150
MAX_PER_SCOPE = 20
MAX_TOTAL = 200
MAX_FORECAST_BYTES = 64 * 1024
MAX_ACTUALS_BYTES = 16 * 1024
MAX_ACTUALS_REQUEST_BYTES = MAX_ACTUALS_BYTES + 64 * 1024
MAX_ACTUAL_MAGNITUDE = 2 ** 53 - 1
MAX_ACTUAL_CORRECTIONS = 90
MAX_CORRECTION_REASON_LENGTH = 300
MAX_SCORED_PREDICTION_MAGNITUDE = 1e150  # Leave room for 90 squared errors in float64.
DECIMAL_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')


class ForecastOutcomeError(ValueError):
    """Input cannot be accepted under the frozen forecast contract."""


class ForecastActualsTooLarge(ForecastOutcomeError):
    """Actuals upload exceeds the web budget."""


class ForecastCorrectionError(ForecastOutcomeError):
    def __init__(self, message, *, field=None, status=422):
        super().__init__(message)
        self.field = field
        self.status = status


def _parse_actual(value):
    value = value.strip()
    if len(value) > 100 or not DECIMAL_NUMBER.fullmatch(value):
        raise ForecastOutcomeError('Actuals must be finite decimal numbers.')
    try:
        precise = Decimal(value)
        number = float(precise)
        if precise.copy_abs() > MAX_ACTUAL_MAGNITUDE or (precise and not number):
            raise ForecastOutcomeError('An actual is outside the supported numeric range.')
        return number
    except InvalidOperation:
        raise ForecastOutcomeError('An actual is outside the supported numeric range.') from None


def correction_from_form(form):
    dates = form.getlist('correction_date')
    if len(dates) != 1 or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}:[1-9][0-9]{0,2}', dates[0]):
        raise ForecastCorrectionError('Select one saved actual to correct.', field='date')
    day, revision = dates[0].split(':')
    try:
        validate_forecast_date(day)
    except ValueError:
        raise ForecastCorrectionError('Select one saved actual to correct.', field='date') from None
    values = form.getlist('correction_value')
    if len(values) != 1:
        raise ForecastCorrectionError('Enter one corrected value.', field='value')
    try:
        value = _parse_actual(values[0])
    except ForecastOutcomeError as error:
        raise ForecastCorrectionError(str(error), field='value') from None
    reasons = form.getlist('correction_reason')
    reason = reasons[0].strip() if len(reasons) == 1 else ''
    if not 1 <= len(reason) <= MAX_CORRECTION_REASON_LENGTH or not reason.isprintable():
        raise ForecastCorrectionError('Enter a reason using 1–300 printable characters.', field='reason')
    return day, int(revision), value, reason


def load_actuals(upload):
    """Read an exact two-column daily facts file without dataframe inference."""
    if not (upload.filename or '').lower().endswith('.csv'):
        raise ForecastOutcomeError('Choose a UTF-8 CSV file.')
    raw = upload.stream.read(MAX_ACTUALS_BYTES + 1)
    if len(raw) > MAX_ACTUALS_BYTES:
        raise ForecastActualsTooLarge('Use an actuals CSV of at most 16 KiB.')
    try:
        records = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
        if next(records, None) != ['date', 'actual']:
            raise ForecastOutcomeError('Use exactly two columns with the header date,actual.')
        actuals = {}
        for row in records:
            if len(row) != 2 or len(actuals) >= 90:
                raise ForecastOutcomeError('Provide 1–90 rows with a date and actual value.')
            try:
                day = validate_forecast_date(row[0]).isoformat()
            except ValueError:
                raise ForecastOutcomeError('Use dates in YYYY-MM-DD format.') from None
            if day in actuals:
                raise ForecastOutcomeError('Provide each actual date only once per upload.')
            actuals[day] = _parse_actual(row[1])
    except (UnicodeError, csv.Error):
        raise ForecastOutcomeError('Use a valid comma-delimited UTF-8 CSV.') from None
    if not actuals:
        raise ForecastOutcomeError('Include at least one daily actual.')
    return actuals


def store_forecast(connection, scope, report, artifact, now):
    if validate_forecast_date(report['forecast_origin']) > now.date():
        raise ForecastOutcomeError('Choose an issue date on or before today in UTC to save a forecast.')
    if report['status'] != 'ok':
        return None, False
    if any(abs(point[key]) > MAX_SCORED_PREDICTION_MAGNITUDE
           for point in report['predictions'] for key in ('prediction', 'baseline')):
        raise ForecastOutcomeError('Forecast values exceed the supported scoring range. Run without saving.')
    stamp = now.isoformat(timespec='microseconds')
    connection.execute('DELETE FROM forecast_outcomes WHERE expires_at<=?', (stamp,))
    existing = connection.execute('''SELECT prediction_id, forecast_json FROM forecast_outcomes
        WHERE scope_id=? AND model_id=? AND forecast_origin=?''',
        (scope, report['model_id'], report['forecast_origin'])).fetchone()
    policy = artifact['decision_policy']
    weights = None if policy is None else {
        key: policy[key] for key in ('underprediction_weight', 'overprediction_weight')
    }
    payload = {
        'report': {key: value for key, value in report.items() if key != 'prediction_id'},
        'target_column': artifact['task']['target_column'],
        'target_unit': artifact['target_unit'],
        'reporting_delay_days': artifact['task'].get('reporting_delay_days', 0),
        'error_weights': weights,
    }
    maximum_mae = policy['maximum_mae'] if policy is not None else None
    if existing is not None:
        saved = json.loads(existing['forecast_json'])
        # A replay of a pre-limit record keeps its original snapshot.
        if 'maximum_mae' in saved:
            payload['maximum_mae'] = maximum_mae
        if saved != payload:
            raise ForecastOutcomeError('A different forecast from this model already exists for this issue date.')
        return existing['prediction_id'], False
    if maximum_mae is not None:
        payload['maximum_mae'] = maximum_mae
    serialized = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(serialized.encode('utf-8')) > MAX_FORECAST_BYTES:
        raise ForecastOutcomeError('The forecast is too large to save. Run without saving.')
    total, owned = connection.execute(
        'SELECT count(*), count(CASE WHEN scope_id=? THEN 1 END) FROM forecast_outcomes',
        (scope,)).fetchone()
    if total >= MAX_TOTAL or owned >= MAX_PER_SCOPE:
        raise ForecastOutcomeError('Saved forecast capacity reached. Run without saving or wait for a record to expire.')
    prediction_id = report['prediction_id']
    connection.execute('''INSERT INTO forecast_outcomes
        (scope_id, prediction_id, model_id, forecast_origin, registered_at, expires_at, forecast_json, actuals_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, '{}')''',
        (scope, prediction_id, report['model_id'], report['forecast_origin'], stamp,
         (now + timedelta(days=RETENTION_DAYS)).isoformat(timespec='microseconds'), serialized))
    return prediction_id, True


def forecast_evaluation_timing(registered_at, forecast_origin):
    """Classify using the first server save and the declared UTC issue day."""
    try:
        saved = datetime.fromisoformat(registered_at)
        origin = validate_forecast_date(forecast_origin)
        if saved.utcoffset() is None:
            return 'unverified'
        saved_day = saved.astimezone(timezone.utc).date()
    except (TypeError, ValueError, OverflowError):
        return 'unverified'
    if saved_day == origin:
        return 'prospective'
    return 'retrospective' if saved_day > origin else 'unverified'


def forecast_view(row, payload, actuals):
    report = payload['report']
    points = []
    corrections = []
    for point in report['predictions']:
        actual = actuals.get(point['date'], {})
        prior = actual.get('superseded', [])
        points.append(dict(point, actual=actual.get('value'), revision=len(prior) + 1))
        if prior:
            corrections.append({
                'date': point['date'],
                'versions': [
                    {
                        'value': item['value'], 'received_at': item['received_at'], 'reason': item.get('reason'),
                        'model_error': abs(item['value'] - point['prediction']),
                        'baseline_error': abs(item['value'] - point['baseline']),
                    }
                    for item in [*prior, actual]
                ],
            })
    matched = [point for point in points if point['actual'] is not None]
    model_scores = baseline_scores = None
    if matched:
        observed = [point['actual'] for point in matched]
        weights = payload['error_weights']
        model_scores = score_predictions(observed, [point['prediction'] for point in matched],
                                         decision_policy=weights)
        baseline_scores = score_predictions(observed, [point['baseline'] for point in matched],
                                            decision_policy=weights)
    return {
        'prediction_id': row['prediction_id'],
        'registered_at': row['registered_at'],
        'expires_at': row['expires_at'],
        'forecast_origin': report['forecast_origin'],
        'evaluation_timing': forecast_evaluation_timing(row['registered_at'], report['forecast_origin']),
        'target_column': payload['target_column'],
        'target_unit': payload['target_unit'],
        'reporting_delay_days': payload['reporting_delay_days'],
        'model_name': report['model_name'],
        'baseline_name': report['baseline_name'],
        'points': points,
        'corrections': corrections,
        'matched_count': len(matched),
        'model_scores': model_scores,
        'baseline_scores': baseline_scores,
    }


def merge_actuals(payload, recorded, actuals, now):
    points = {point['date'] for point in payload['report']['predictions']}
    latest_available = now.date() - timedelta(days=payload['reporting_delay_days'])
    stamp = now.isoformat(timespec='microseconds')
    merged = recorded.copy()
    for day, value in actuals.items():
        if day not in points:
            raise ForecastOutcomeError('Every actual date must belong to this forecast’s horizon.')
        if validate_forecast_date(day) > latest_available:
            raise ForecastOutcomeError('An actual is too early under this forecast’s reporting delay.')
        if payload['target_unit'] == 'count' and value < 0:
            raise ForecastOutcomeError('Count actuals cannot be negative.')
        if day in merged and merged[day]['value'] != value:
            raise ForecastOutcomeError('An uploaded value conflicts with a saved actual. The batch was not applied.')
        merged.setdefault(day, {'value': value, 'received_at': stamp})
    return merged


def correct_actual(payload, recorded, day, expected_revision, value, reason, *, now, correction_count=None):
    """Append a revision only if the owner saw the latest version."""
    current = recorded.get(day)
    if current is None:
        raise ForecastCorrectionError('Select a date with a saved actual.', field='date')
    if validate_forecast_date(day) > now.date() - timedelta(days=payload['reporting_delay_days']):
        raise ForecastCorrectionError('This actual is too early under the reporting delay.', field='date')
    if payload['target_unit'] == 'count' and value < 0:
        raise ForecastCorrectionError('Count actuals cannot be negative.', field='value')
    prior = current.get('superseded', [])
    current_revision = len(prior) + 1
    if expected_revision != current_revision:
        # The same correction can be retried after a lost response.
        if (expected_revision == current_revision - 1 and expected_revision >= 1
                and current['value'] == value and current.get('reason') == reason):
            return recorded
        raise ForecastCorrectionError('This actual has changed. Reload the forecast before correcting it.',
                                      field='date', status=409)
    if current['value'] == value:
        raise ForecastCorrectionError('Enter a value different from the current actual.', field='value')
    if correction_count is None:
        correction_count = sum(len(item.get('superseded', [])) for item in recorded.values())
    if correction_count >= MAX_ACTUAL_CORRECTIONS:
        raise ForecastCorrectionError('The correction limit for these actuals has been reached.')
    return recorded | {
        day: {
            'value': value,
            'received_at': now.isoformat(timespec='microseconds'),
            'reason': reason,
            'superseded': [*prior, {key: val for key, val in current.items() if key != 'superseded'}],
        },
    }
