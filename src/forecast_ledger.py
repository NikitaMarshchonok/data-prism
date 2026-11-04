"""Opt-in local forecast/outcome journal, not a hosted registry or live monitor.

The local operator controls this data-bearing directory. SQLite transactions
preserve frozen predictions and explicit revisions; they do not authenticate
the operator, prove historical issuance, or verify source release timestamps.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3

from src.forecast_artifact import predict_artifact, validate_artifact
from src.forecast_policy import evaluate_forecast_policy
from src.forecast_scoring import score_predictions
from src.forecasting import validate_forecast_date


CONTRACT = 'local-forecast-outcome-ledger-v1'
MAX_FORECASTS = 1000
MAX_OUTCOME_REVISIONS = 100000
DATABASE = 'forecasts.sqlite3'


def _now():
    return datetime.now(timezone.utc)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _identifier(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('A valid prediction ID is required.')
    return value


def _actual(value, unit):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (type(value) is int and abs(value) > 2 ** 53 - 1):
        raise ValueError('Actuals must be finite numbers with safe integer precision.')
    if unit == 'count' and value < 0:
        raise ValueError('Count actuals cannot be negative.')
    return float(value)


class ForecastLedger:
    """One private local operator, explicit creation, no automatic web storage."""

    @classmethod
    def create(cls, directory):
        directory = Path(directory).expanduser()
        # Never adopt or overwrite an existing directory, database or symlink.
        directory.mkdir(mode=0o700)
        path = directory / DATABASE
        created_file = False
        connection = None
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            created_file = True
            os.close(descriptor)
            connection = sqlite3.connect(path)
            with connection:
                connection.executescript('''
                    PRAGMA foreign_keys=ON;
                    CREATE TABLE ledger_metadata (contract TEXT NOT NULL);
                    INSERT INTO ledger_metadata VALUES ('local-forecast-outcome-ledger-v1');
                    CREATE TABLE forecasts (
                        id TEXT PRIMARY KEY, model_sha256 TEXT NOT NULL,
                        registered_at TEXT NOT NULL, payload TEXT NOT NULL,
                        payload_sha256 TEXT NOT NULL
                    );
                    CREATE TABLE points (
                        prediction_id TEXT NOT NULL REFERENCES forecasts(id) ON DELETE CASCADE,
                        date TEXT NOT NULL, lead INTEGER NOT NULL,
                        prediction REAL NOT NULL, baseline REAL NOT NULL,
                        PRIMARY KEY(prediction_id, date)
                    );
                    CREATE TABLE outcomes (
                        model_sha256 TEXT NOT NULL, date TEXT NOT NULL,
                        revision INTEGER NOT NULL, actual REAL NOT NULL,
                        received_at TEXT NOT NULL, as_of_date TEXT NOT NULL,
                        reason TEXT,
                        PRIMARY KEY(model_sha256, date, revision)
                    );
                    PRAGMA user_version=1;
                ''')
        except Exception:
            # Initialization owns only these newly created paths.
            if connection is not None:
                connection.close()
            if created_file:
                path.unlink()
            directory.rmdir()
            raise
        finally:
            if connection is not None:
                connection.close()
        return cls(directory)

    def __init__(self, directory):
        self.directory = Path(directory).expanduser()
        self.path = self.directory / DATABASE
        if (self.directory.is_symlink() or not self.directory.is_dir()
                or self.path.is_symlink() or not self.path.is_file()):
            raise ValueError('Initialize a private local forecast ledger first.')
        if self.directory.stat().st_mode & 0o077 or self.path.stat().st_mode & 0o077:
            raise ValueError('Ledger directory/file must be private (0700/0600).')
        with self._connection() as connection:
            if connection.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise ValueError('Unsupported local ledger version.')
            if [row[0] for row in connection.execute('SELECT contract FROM ledger_metadata')] != [CONTRACT]:
                raise ValueError('Unsupported local ledger contract.')

    @contextmanager
    def _connection(self):
        # mode=rw prevents silently recreating a missing database on a read.
        connection = sqlite3.connect(self.path.resolve().as_uri() + '?mode=rw', uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('PRAGMA secure_delete=ON')
            with connection:
                yield connection
        finally:
            connection.close()

    def issue(self, artifact, *, history=None, forecast_origin=None):
        """Compute with frozen parameters and atomically retain exactly that output."""
        validate_artifact(artifact)
        report = predict_artifact(artifact, history=history, forecast_origin=forecast_origin)
        now = _now()
        if validate_forecast_date(report['forecast_origin']) > now.date():
            raise ValueError('A local journal cannot register a future issue date.')
        payload = {'forecast': report, 'target_unit': artifact['target_unit'],
                   'task': artifact['task'], 'decision_policy': artifact['decision_policy']}
        serialized = _json(payload)
        timestamp = now.isoformat()
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if connection.execute('SELECT count(*) FROM forecasts').fetchone()[0] >= MAX_FORECASTS:
                raise ValueError('Local forecast capacity reached; review retention explicitly.')
            connection.execute('INSERT INTO forecasts VALUES (?, ?, ?, ?, ?)',
                               (report['prediction_id'], report['model_sha256'], timestamp,
                                serialized, hashlib.sha256(serialized.encode('utf-8')).hexdigest()))
            connection.executemany('INSERT INTO points VALUES (?, ?, ?, ?, ?)',
                                   [(report['prediction_id'], point['date'], lead,
                                     point['prediction'], point['baseline'])
                                    for lead, point in enumerate(report['predictions'], 1)])
        return report | {'ledger': {'contract': CONTRACT, 'status': 'recorded',
                                   'registered_at': timestamp,
                                   'historical_issuance_verified': False}}

    @staticmethod
    def _load(connection, prediction_id):
        row = connection.execute('SELECT * FROM forecasts WHERE id=?',
                                 (_identifier(prediction_id),)).fetchone()
        if row is None:
            raise ValueError('Prediction is not in this local ledger.')
        digest = hashlib.sha256(row['payload'].encode('utf-8')).hexdigest()
        if digest != row['payload_sha256']:
            raise ValueError('Stored forecast integrity check failed.')
        payload = json.loads(row['payload'])
        forecast = payload['forecast']
        if forecast['prediction_id'] != prediction_id or forecast['model_sha256'] != row['model_sha256']:
            raise ValueError('Stored forecast identity disagrees.')
        points = connection.execute('SELECT * FROM points WHERE prediction_id=? ORDER BY lead',
                                    (prediction_id,)).fetchall()
        expected = [(p['date'], p['prediction'], p['baseline']) for p in forecast['predictions']]
        if ([(p['date'], p['prediction'], p['baseline']) for p in points] != expected
                or [p['lead'] for p in points] != list(range(1, len(expected) + 1))):
            raise ValueError('Stored prediction points disagree with the frozen record.')
        return row, payload, points

    def accept_actuals(self, prediction_id, actuals, *, as_of_date, correction_reason=None):
        """Atomic bounded batch; duplicate facts are no-ops, revisions are explicit.

        Facts are shared only across forecasts of the exact same model SHA/date.
        The as-of date is an operator assertion, capped by today's UTC date.
        """
        as_of = validate_forecast_date(as_of_date)
        now = _now()
        if as_of > now.date():
            raise ValueError('Actuals cannot be declared available in the future.')
        if type(actuals) is not list or not 1 <= len(actuals) <= 90:
            raise ValueError('Provide 1–90 explicit daily actuals.')
        if correction_reason is not None:
            if (type(correction_reason) is not str or not 1 <= len(correction_reason.strip()) <= 500):
                raise ValueError('A correction reason must contain 1–500 characters.')
            correction_reason = correction_reason.strip()
            correction_reason.encode('utf-8')
        changed = unchanged = 0
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            stored, payload, points = self._load(connection, prediction_id)
            dates = {point['date'] for point in points}
            delay = payload['task']['reporting_delay_days']
            seen = set()
            for item in actuals:
                if type(item) is not dict or set(item) != {'date', 'actual'}:
                    raise ValueError('Actual rows require exactly date and actual.')
                day = validate_forecast_date(item['date'])
                if item['date'] not in dates or item['date'] in seen:
                    raise ValueError('Actual dates must be unique members of the issued horizon.')
                seen.add(item['date'])
                if day + timedelta(days=delay) > as_of:
                    raise ValueError('An actual is not yet available under the fixed release delay.')
                value = _actual(item['actual'], payload['target_unit'])
                previous = connection.execute('''SELECT * FROM outcomes
                    WHERE model_sha256=? AND date=? ORDER BY revision DESC LIMIT 1''',
                    (stored['model_sha256'], item['date'])).fetchone()
                if previous is not None:
                    if as_of_date < previous['as_of_date']:
                        raise ValueError('Outcome updates cannot move availability backwards.')
                    if previous['actual'] == value:
                        unchanged += 1
                        continue
                    if correction_reason is None:
                        raise ValueError('A changed actual requires an explicit correction reason.')
                if connection.execute('SELECT count(*) FROM outcomes').fetchone()[0] >= MAX_OUTCOME_REVISIONS:
                    raise ValueError('Local outcome capacity reached; review retention explicitly.')
                connection.execute('INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?)',
                                   (stored['model_sha256'], item['date'],
                                    previous['revision'] + 1 if previous else 1, value,
                                    now.isoformat(), as_of_date, correction_reason))
                changed += 1
        return {'contract': CONTRACT, 'status': 'accepted', 'changed': changed, 'unchanged': unchanged}

    def evaluate(self, prediction_id):
        """Paired measured errors; partial coverage never becomes horizon success."""
        with self._connection() as connection:
            # Keep all point/outcome reads in a consistent SQLite snapshot.
            connection.execute('BEGIN')
            stored, payload, points = self._load(connection, prediction_id)
            matched, pending = [], []
            for point in points:
                outcome = connection.execute('''SELECT * FROM outcomes
                    WHERE model_sha256=? AND date=? ORDER BY revision DESC LIMIT 1''',
                    (stored['model_sha256'], point['date'])).fetchone()
                if outcome is None:
                    pending.append(point['date'])
                else:
                    actual = _actual(outcome['actual'], payload['target_unit'])
                    matched.append({'date': point['date'], 'lead': point['lead'], 'actual': actual,
                                    'prediction': point['prediction'], 'baseline': point['baseline'],
                                    'outcome_revision': outcome['revision'],
                                    'as_of_date': outcome['as_of_date'], 'received_at': outcome['received_at'],
                                    'absolute_error': abs(actual - point['prediction']),
                                    'baseline_absolute_error': abs(actual - point['baseline'])})
        forecast, policy = payload['forecast'], payload['decision_policy']
        metrics = baseline_metrics = improvement = business = None
        if matched:
            observed = [row['actual'] for row in matched]
            metrics = score_predictions(observed, [row['prediction'] for row in matched], decision_policy=policy)
            baseline_metrics = score_predictions(observed, [row['baseline'] for row in matched], decision_policy=policy)
            if baseline_metrics['mae'] > 0:
                improvement = 1 - metrics['mae'] / baseline_metrics['mae']
                if not math.isfinite(improvement):
                    improvement = None
            if policy is not None and not pending:
                evaluation = evaluate_forecast_policy(policy, metrics, baseline_metrics)
                checks = evaluation['checks'] | {'original_forecast_domain_valid': forecast['status'] == 'ok'}
                passed = all(checks.values())
                business = {'status': 'criteria_met_on_reported_outcomes' if passed else 'criteria_not_met_on_reported_outcomes',
                            'checks': checks,
                            'relative_improvement': evaluation['relative_improvement']}
        return {'contract': CONTRACT, 'prediction_id': prediction_id,
                'model_id': forecast['model_id'], 'model_sha256': forecast['model_sha256'],
                'registered_at': stored['registered_at'], 'forecast_origin': forecast['forecast_origin'],
                'model_name': forecast['model_name'], 'baseline_name': forecast['baseline_name'],
                'target_column': payload['task']['target_column'], 'time_column': payload['task']['time_column'],
                'target_unit': payload['target_unit'],
                'reporting_delay_days': payload['task']['reporting_delay_days'],
                'metrics_scope': 'matched_dates_only',
                'forecast_status': forecast['status'],
                'quality_status': 'unknown' if not matched else ('partial' if pending else 'measured_on_reported_outcomes'),
                'expected_points': len(points), 'matched_points': len(matched),
                'metrics': metrics, 'baseline_metrics': baseline_metrics,
                'mae_relative_improvement': improvement, 'business_evaluation': business,
                'matched': matched, 'pending_dates': pending,
                'historical_issuance_verified': False, 'production_readiness': 'not_established',
                'financial_value': 'not_assessed',
                'limitations': ['Local operator assertions; historical issuance and actual release timestamps are not verified.',
                                'Partial errors use only the same matched dates for model and frozen baseline.',
                                'Overlapping forecasts share facts for the same model hash; errors are not independent.',
                                'No automatic alerts, drift inference, retraining, promotion or hosted durability.']}

    def revisions(self, prediction_id):
        """Only revisions for the named forecast's dates, including superseded facts."""
        with self._connection() as connection:
            connection.execute('BEGIN')
            stored, _, _ = self._load(connection, prediction_id)
            rows = connection.execute('''SELECT o.date, o.revision, o.actual,
                    o.received_at, o.as_of_date, o.reason FROM outcomes o
                JOIN points p ON p.date=o.date AND p.prediction_id=?
                WHERE o.model_sha256=? ORDER BY o.date, o.revision''',
                (prediction_id, stored['model_sha256'])).fetchall()
        return [dict(row) for row in rows]

    def delete(self, prediction_id):
        """Explicit logical deletion; shared facts remain while another run uses them."""
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            stored, _, points = self._load(connection, prediction_id)
            connection.execute('DELETE FROM forecasts WHERE id=?', (prediction_id,))
            deleted = 0
            for point in points:
                cursor = connection.execute('''DELETE FROM outcomes
                    WHERE model_sha256=? AND date=? AND NOT EXISTS (
                        SELECT 1 FROM points p JOIN forecasts f ON f.id=p.prediction_id
                        WHERE f.model_sha256=? AND p.date=?)''',
                    (stored['model_sha256'], point['date'], stored['model_sha256'], point['date']))
                deleted += cursor.rowcount
        return {'contract': CONTRACT, 'status': 'deleted', 'prediction_id': prediction_id,
                'outcome_revisions_deleted': deleted}
