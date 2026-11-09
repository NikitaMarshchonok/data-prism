"""Explicit account-owned identity for shared forecast actuals."""
from datetime import timedelta
import json
import re
import unicodedata
import uuid

from src.forecast_scoring import score_predictions
from .forecast_outcomes import ForecastOutcomeError, forecast_evaluation_timing


class ForecastSeriesError(ForecastOutcomeError):
    """The requested series binding cannot be used for this forecast."""


def initialize_series(connection):
    connection.execute('''CREATE TABLE IF NOT EXISTS forecast_series (
        scope_id TEXT NOT NULL, series_id TEXT NOT NULL,
        label TEXT NOT NULL, label_key TEXT NOT NULL,
        time_column TEXT NOT NULL, target_column TEXT NOT NULL,
        target_unit TEXT NOT NULL, reporting_delay_days INTEGER NOT NULL,
        PRIMARY KEY(scope_id, series_id), UNIQUE(scope_id, label_key)
    )''')
    connection.execute('''CREATE TABLE IF NOT EXISTS forecast_series_links (
        scope_id TEXT NOT NULL, prediction_id TEXT NOT NULL, series_id TEXT NOT NULL,
        first_day TEXT NOT NULL, last_day TEXT NOT NULL,
        PRIMARY KEY(scope_id, prediction_id)
    )''')
    connection.execute('''CREATE INDEX IF NOT EXISTS idx_forecast_series_links_series
        ON forecast_series_links(scope_id, series_id)''')
    connection.execute('''CREATE TABLE IF NOT EXISTS forecast_series_actuals (
        scope_id TEXT NOT NULL, series_id TEXT NOT NULL, day TEXT NOT NULL,
        actual_json TEXT NOT NULL, correction_count INTEGER NOT NULL,
        PRIMARY KEY(scope_id, series_id, day)
    )''')
    # A shared fact lives only while some retained forecast still covers it.
    # Keeping cleanup on the parent DELETE also covers expiry and account
    # erasure through an older application version using this same database.
    connection.execute('''CREATE TRIGGER IF NOT EXISTS forecast_series_cleanup
        AFTER DELETE ON forecast_outcomes BEGIN
            DELETE FROM forecast_series_actuals
            WHERE scope_id=OLD.scope_id
              AND series_id=(SELECT series_id FROM forecast_series_links
                             WHERE scope_id=OLD.scope_id AND prediction_id=OLD.prediction_id)
              AND NOT EXISTS (
                  SELECT 1 FROM forecast_series_links AS link
                  WHERE link.scope_id=forecast_series_actuals.scope_id
                    AND link.series_id=forecast_series_actuals.series_id
                    AND link.prediction_id<>OLD.prediction_id
                    AND forecast_series_actuals.day BETWEEN link.first_day AND link.last_day
              );
            DELETE FROM forecast_series_links
            WHERE scope_id=OLD.scope_id AND prediction_id=OLD.prediction_id;
            DELETE FROM forecast_series
            WHERE scope_id=OLD.scope_id AND NOT EXISTS (
                SELECT 1 FROM forecast_series_links AS link
                WHERE link.scope_id=forecast_series.scope_id AND link.series_id=forecast_series.series_id
            );
        END''')
    connection.execute('''CREATE TRIGGER IF NOT EXISTS forecast_series_no_local_actuals
        BEFORE UPDATE OF actuals_json ON forecast_outcomes
        WHEN OLD.actuals_json<>NEW.actuals_json AND EXISTS (
            SELECT 1 FROM forecast_series_links
            WHERE scope_id=OLD.scope_id AND prediction_id=OLD.prediction_id
        ) BEGIN
            SELECT RAISE(ABORT, 'Actuals are stored in the linked series.');
        END''')


def series_from_form(form, *, retain):
    selected = form.getlist('series_id')
    labels = form.getlist('new_series_label')
    confirmation = form.getlist('series_confirmed')
    if len(selected) > 1 or len(labels) > 1 or confirmation not in ([], ['yes']):
        raise ForecastSeriesError('Choose one series and use the confirmation checkbox.')
    series_id = selected[0] if selected else ''
    label = unicodedata.normalize('NFC', labels[0].strip()) if labels else ''
    if not (series_id or label or confirmation):
        return None
    if not retain or not confirmation:
        raise ForecastSeriesError('To share actuals, choose to save the forecast and confirm the series.')
    if bool(series_id) == bool(label):
        raise ForecastSeriesError('Choose an existing series or enter one new name.')
    if series_id and not re.fullmatch('[0-9a-f]{32}', series_id):
        raise ForecastSeriesError('The selected series is unavailable or incompatible.')
    if label and (len(label) > 80 or not label.isprintable()):
        raise ForecastSeriesError('Enter a series name using 1–80 printable characters.')
    return series_id, label


def list_series(connection, scope, now):
    return [dict(row) for row in connection.execute('''
        SELECT series_id, label, time_column, target_column, target_unit, reporting_delay_days
        FROM forecast_series AS series WHERE scope_id=? AND EXISTS (
            SELECT 1 FROM forecast_series_links AS link JOIN forecast_outcomes AS outcome
              ON outcome.scope_id=link.scope_id AND outcome.prediction_id=link.prediction_id
            WHERE link.scope_id=series.scope_id AND link.series_id=series.series_id AND outcome.expires_at>?
        ) ORDER BY label_key, series_id''', (scope, now))]


def get_link(connection, scope, prediction_id):
    return connection.execute('''SELECT link.series_id, series.label, link.first_day, link.last_day
        FROM forecast_series_links AS link JOIN forecast_series AS series
          ON series.scope_id=link.scope_id AND series.series_id=link.series_id
        WHERE link.scope_id=? AND link.prediction_id=?''', (scope, prediction_id)).fetchone()


def bind_series(connection, scope, prediction_id, *, created, choice, artifact, points):
    current = get_link(connection, scope, prediction_id)
    if choice is None:
        if current is not None:
            raise ForecastSeriesError('This forecast was saved with a shared series. Select that same series to retry.')
        return

    series_id, label = choice
    task = artifact['task']
    contract = (task['time_column'], task['target_column'], artifact['target_unit'],
                task.get('reporting_delay_days', 0))
    if series_id:
        series = connection.execute('SELECT * FROM forecast_series WHERE scope_id=? AND series_id=?',
                                    (scope, series_id)).fetchone()
        if series is None or (
            series['time_column'], series['target_column'], series['target_unit'], series['reporting_delay_days'],
        ) != contract:
            raise ForecastSeriesError('The selected series is unavailable or incompatible.')
    else:
        label_key = label.casefold()
        series = connection.execute('SELECT series_id FROM forecast_series WHERE scope_id=? AND label_key=?',
                                    (scope, label_key)).fetchone()
        if series is not None:
            if current is None or series['series_id'] != current['series_id']:
                raise ForecastSeriesError('A series with that name already exists. Choose it from the list.')
            return
        if created:
            series_id = uuid.uuid4().hex
            connection.execute('INSERT INTO forecast_series VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                               (scope, series_id, label, label_key, *contract))

    if not created:
        if current is None or current['series_id'] != series_id:
            raise ForecastSeriesError('The series of a saved forecast cannot be changed.')
        return
    connection.execute('INSERT INTO forecast_series_links VALUES (?, ?, ?, ?, ?)',
                       (scope, prediction_id, series_id, points[0]['date'], points[-1]['date']))


def read_series_actuals(connection, scope, link):
    return {row['day']: json.loads(row['actual_json']) for row in connection.execute('''
        SELECT day, actual_json FROM forecast_series_actuals
        WHERE scope_id=? AND series_id=? AND day BETWEEN ? AND ?''',
        (scope, link['series_id'], link['first_day'], link['last_day']))}


def evaluate_series_model(connection, scope, series_id, report, *, now, reporting_delay_days):
    """Compare one frozen model and baseline by lead across prospective issues."""
    rows = connection.execute('''
        SELECT outcome.forecast_origin, outcome.registered_at, outcome.forecast_json,
               link.first_day, link.last_day
        FROM forecast_series_links AS link JOIN forecast_outcomes AS outcome
          ON outcome.scope_id=link.scope_id AND outcome.prediction_id=link.prediction_id
        WHERE link.scope_id=? AND link.series_id=? AND outcome.model_id=? AND outcome.expires_at>?
        ORDER BY outcome.forecast_origin''',
        (scope, series_id, report['model_id'], now.isoformat(timespec='microseconds')))
    eligible = []
    for row in rows:
        if forecast_evaluation_timing(row['registered_at'], row['forecast_origin']) != 'prospective':
            continue
        saved = json.loads(row['forecast_json'])['report']
        if saved['model_sha256'] == report['model_sha256']:
            eligible.append((row, saved['predictions']))
    if not eligible:
        return {'forecast_count': 0, 'leads': []}

    actuals = read_series_actuals(connection, scope, {
        'series_id': series_id,
        'first_day': min(row['first_day'] for row, _ in eligible),
        'last_day': max(row['last_day'] for row, _ in eligible),
    })
    due_through = (now.date() - timedelta(days=reporting_delay_days)).isoformat()
    leads = []
    for lead in range(len(report['predictions'])):
        # A model has one retained forecast per issue date, so each target
        # date can appear once per lead, even if the horizons overlap.
        due = [points[lead] for _, points in eligible if points[lead]['date'] <= due_through]
        matched = [point for point in due if point['date'] in actuals]
        model_mae = baseline_mae = None
        if matched:
            observed = [actuals[point['date']]['value'] for point in matched]
            model_mae = score_predictions(observed, [point['prediction'] for point in matched])['mae']
            baseline_mae = score_predictions(observed, [point['baseline'] for point in matched])['mae']
        leads.append({
            'lead_days': lead + 1, 'due_count': len(due), 'matched_count': len(matched),
            'model_mae': model_mae, 'baseline_mae': baseline_mae,
        })
    return {'forecast_count': len(eligible), 'leads': leads}


def count_series_corrections(connection, scope, series_id):
    return connection.execute('''SELECT coalesce(sum(correction_count), 0)
        FROM forecast_series_actuals WHERE scope_id=? AND series_id=?''', (scope, series_id)).fetchone()[0]


def write_series_actuals(connection, scope, series_id, before, after):
    connection.executemany('''INSERT INTO forecast_series_actuals
        (scope_id, series_id, day, actual_json, correction_count) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(scope_id, series_id, day) DO UPDATE SET
          actual_json=excluded.actual_json, correction_count=excluded.correction_count''', [
        (scope, series_id, day, json.dumps(actual, ensure_ascii=False, allow_nan=False),
         len(actual.get('superseded', [])))
        for day, actual in after.items() if before.get(day) != actual
    ])
