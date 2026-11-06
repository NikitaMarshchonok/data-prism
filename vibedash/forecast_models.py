"""Owner-scoped storage for frozen forecast models.

Uses the existing analysis database and deletion fence. Persistent infrastructure
is operator-declared and mount continuity checked, not physically proven here.
No imported Python objects, implicit saving, activation or retraining.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from src.forecast_artifact import MAX_ARTIFACT_BYTES, predict_artifact, validate_artifact
from src.runtime_storage import evaluate_runtime_storage_contract
from .analysis_jobs import AnalysisJobStore, _validated_job_id, _validated_scope_id
from .forecast_outcomes import correct_actual, forecast_view, merge_actuals, store_forecast
from . import forecast_series


CONTRACT = 'scoped-forecast-model-store-v1'
RETENTION_CONTRACT = 'forecast-model-retention-v1'
MAX_MODELS_PER_SCOPE = 10
MAX_MODELS_TOTAL = 100
MAX_BYTES_PER_SCOPE = 16 * 1024 * 1024
MAX_BYTES_TOTAL = 128 * 1024 * 1024


class ForecastModelStorageUnavailable(ValueError):
    """The persistent-state declaration or continuity check is not satisfied."""


class ForecastModelCapacityError(ValueError):
    """Explicit capacity is exhausted; no retained model is silently removed."""


class ForecastModelNotFound(ValueError):
    """No unexpired model exists in the authenticated scope."""


def _now():
    return datetime.now(timezone.utc)


def _timestamp(value):
    return value.isoformat(timespec='microseconds')


def _serialized(artifact):
    validate_artifact(artifact)
    return json.dumps(artifact, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def storage_settings(config):
    """Resolve opt-in web storage without creating a database or a registry."""
    if (config.get('VIBEDASH_MODEL_STORAGE_ENABLED') is not True
            or config.get('DATA_PRISM_STATE_DIRECTORY_CONFIGURED') is not True):
        return None
    root = config.get('DATA_PRISM_STATE_DIR')
    database = config.get('VIBEDASH_JOB_STORE_PATH')
    if not root or not database:
        return None
    root = Path(root).expanduser().absolute()
    database = Path(database).expanduser().absolute()
    if database != root / 'jobs' / 'analysis_jobs.sqlite3':
        return None
    if root.is_symlink() or database.parent.is_symlink() or database.is_symlink():
        return None
    settings = {
        'state_directory': root,
        'deployment_profile': config.get('DATA_PRISM_DEPLOYMENT_PROFILE'),
        'state_durability': config.get('DATA_PRISM_STATE_DURABILITY'),
        'expected_state_id': config.get('DATA_PRISM_EXPECTED_STATE_ID'),
    }
    contract = evaluate_runtime_storage_contract(
        **settings, state_directory_configured=True,
    )
    return settings if contract['production_state_satisfied'] else None


def retention_days(payload):
    """Validate the closed, server-created consent record; absent means no opt-in."""
    if 'forecast_model_retention' not in payload:
        return None
    record = payload['forecast_model_retention']
    if (payload.get('analysis_kind') != 'daily_forecast' or type(record) is not dict
            or set(record) != {'contract', 'days'} or record['contract'] != RETENTION_CONTRACT
            or type(record['days']) is not int or not 1 <= record['days'] <= 30):
        raise ValueError('Invalid forecast model retention record.')
    return record['days']


class ForecastModelStore:
    """Internal server API: scope must come from authenticated server context.

    A scope string alone is not authentication. Web callers must supply
    an authenticated account scope, never a request-supplied or guest scope.
    """

    def __init__(self, state_directory, *, expected_state_id, deployment_profile, state_durability):
        self.state_directory = Path(state_directory).expanduser().absolute()
        self.expected_state_id = expected_state_id
        self.deployment_profile = deployment_profile
        self.state_durability = state_durability
        self.database_path = self.state_directory / 'jobs' / 'analysis_jobs.sqlite3'
        self._check_storage()
        self.jobs = AnalysisJobStore(self.database_path)
        with self.jobs._connection() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS forecast_models (
                scope_id TEXT NOT NULL, model_id TEXT NOT NULL,
                contract TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
                artifact_json TEXT NOT NULL, artifact_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                PRIMARY KEY(scope_id, model_id)
            )''')
            connection.execute('''CREATE INDEX IF NOT EXISTS idx_forecast_models_expiry
                ON forecast_models(scope_id, expires_at)''')
            connection.execute('''CREATE TABLE IF NOT EXISTS forecast_outcomes (
                scope_id TEXT NOT NULL, prediction_id TEXT NOT NULL, model_id TEXT NOT NULL,
                forecast_origin TEXT NOT NULL, registered_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                forecast_json TEXT NOT NULL, actuals_json TEXT NOT NULL,
                PRIMARY KEY(scope_id, prediction_id),
                UNIQUE(scope_id, model_id, forecast_origin)
            )''')
            forecast_series.initialize_series(connection)

    def _check_storage(self):
        contract = evaluate_runtime_storage_contract(
            deployment_profile=self.deployment_profile, state_durability=self.state_durability,
            state_directory_configured=True, state_directory=self.state_directory,
            expected_state_id=self.expected_state_id)
        if not contract['production_state_satisfied']:
            raise ForecastModelStorageUnavailable('Forecast model storage requires verified production state continuity.')
        if (self.state_directory.is_symlink() or (self.state_directory / 'jobs').is_symlink()
                or self.database_path.is_symlink()):
            raise ForecastModelStorageUnavailable('Forecast model state paths cannot be symbolic links.')

    def _guard(self, scope_id, callback):
        scope_id = _validated_scope_id(scope_id)
        self._check_storage()
        if not self.database_path.is_file():
            raise ForecastModelStorageUnavailable('Forecast model state database is unavailable.')
        def guarded(connection):
            # Recheck after acquiring the same fence used by account deletion.
            self._check_storage()
            return callback(connection, scope_id)
        return self.jobs.run_if_scope_open(scope_id, guarded)

    @staticmethod
    def _artifact(row):
        if row['contract'] != CONTRACT:
            raise ValueError('Unsupported stored model contract.')
        serialized = row['artifact_json']
        if (type(serialized) is not str or len(serialized.encode('utf-8')) != row['artifact_bytes']
                or not 0 < row['artifact_bytes'] <= MAX_ARTIFACT_BYTES):
            raise ValueError('Stored model byte budget disagrees.')
        artifact = json.loads(serialized)
        validate_artifact(artifact)
        if artifact['model_id'] != row['model_id'] or artifact['sha256'] != row['artifact_sha256']:
            raise ValueError('Stored model identity disagrees.')
        created, expiry = (datetime.fromisoformat(row[key]) for key in ('created_at', 'expires_at'))
        if (created.tzinfo != timezone.utc or expiry.tzinfo != timezone.utc
                or not timedelta(days=1) <= expiry - created <= timedelta(days=30)):
            raise ValueError('Stored retention dates disagree.')
        return artifact

    def register(self, scope_id, artifact, *, retention_days):
        """Explicit immutable registration; duplicates never extend retention."""
        prepared = self._prepare_registration(artifact, retention_days)
        return self._guard(scope_id, lambda connection, scope:
                           self._register(connection, scope, prepared))

    @staticmethod
    def _prepare_registration(artifact, retention_days):
        if type(retention_days) is not int or not 1 <= retention_days <= 30:
            raise ValueError('Choose an explicit retention of 1–30 days.')
        serialized = _serialized(artifact)
        artifact = json.loads(serialized)  # immutable copy of the validated input
        size = len(serialized.encode('utf-8'))
        if size > MAX_ARTIFACT_BYTES:
            raise ValueError('Model exceeds the artifact byte budget.')
        return artifact, serialized, size, retention_days

    def _register(self, connection, scope, prepared):
        artifact, serialized, size, days = prepared
        now = _now()
        row = connection.execute('SELECT * FROM forecast_models WHERE scope_id=? AND model_id=?',
                                 (scope, artifact['model_id'])).fetchone()
        if row is not None:
            self._artifact(row)
            if row['artifact_sha256'] != artifact['sha256'] or row['artifact_json'] != serialized:
                raise ValueError('A saved model cannot be replaced in place.')
            if row['expires_at'] <= _timestamp(now):
                raise ValueError('An expired model must be purged before explicit registration.')
            return self._summary(row) | {'created': False}
        counts = connection.execute('''SELECT count(*) AS total, coalesce(sum(artifact_bytes), 0) AS bytes,
            coalesce(sum(CASE WHEN scope_id=? THEN 1 ELSE 0 END), 0) AS owned,
            coalesce(sum(CASE WHEN scope_id=? THEN artifact_bytes ELSE 0 END), 0) AS owned_bytes
            FROM forecast_models''', (scope, scope)).fetchone()
        if (counts['total'] >= MAX_MODELS_TOTAL or counts['owned'] >= MAX_MODELS_PER_SCOPE
                or counts['bytes'] + size > MAX_BYTES_TOTAL or counts['owned_bytes'] + size > MAX_BYTES_PER_SCOPE):
            raise ForecastModelCapacityError('Retained model capacity reached; nothing was evicted.')
        connection.execute('INSERT INTO forecast_models VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                           (scope, artifact['model_id'], CONTRACT, artifact['sha256'], serialized,
                            size, _timestamp(now), _timestamp(now + timedelta(days=days))))
        saved = connection.execute('SELECT * FROM forecast_models WHERE scope_id=? AND model_id=?',
                                   (scope, artifact['model_id'])).fetchone()
        return self._summary(saved) | {'created': True}

    def register_completed_job(self, connection, *, job_id, scope_id, artifact):
        """Retain only inside the owned job's completion transaction."""
        job_id, scope = _validated_job_id(job_id), _validated_scope_id(scope_id)
        self._check_storage()
        db = connection.execute('PRAGMA database_list').fetchone()
        if not connection.in_transaction or not db or Path(db[2]).resolve() != self.database_path.resolve():
            raise ValueError('Model registration requires the analysis completion transaction.')
        row = connection.execute('''SELECT payload_json FROM analysis_jobs
            WHERE id=? AND scope_id=? AND status='completed'
              AND NOT EXISTS (SELECT 1 FROM deleted_scopes WHERE scope_id=?)''',
            (job_id, scope, scope)).fetchone()
        payload = json.loads(row['payload_json']) if row is not None else {}
        days = retention_days(payload)
        if days is None:
            raise ValueError('Model registration requires an opted-in, completed forecast job.')
        prepared = self._prepare_registration(artifact, days)
        artifact = prepared[0]
        task = payload['forecast_task']
        if (artifact['model_id'] != job_id or artifact['task'] != task['settings']
                or artifact['target_unit'] != task['target_unit']
                or artifact['decision_policy'] != task.get('decision_policy')):
            raise ValueError('Model does not match the completed forecast task.')
        return self._register(connection, scope, prepared)

    @staticmethod
    def _summary(row):
        return {'contract': CONTRACT, 'model_id': row['model_id'], 'model_sha256': row['artifact_sha256'],
                'created_at': row['created_at'], 'expires_at': row['expires_at'],
                'artifact_bytes': row['artifact_bytes'], 'production_readiness': 'not_established'}

    def _read(self, scope_id, model_id, *, summary_only):
        model_id = _validated_job_id(model_id)
        def read(connection, scope):
            row = connection.execute('''SELECT * FROM forecast_models
                WHERE scope_id=? AND model_id=? AND expires_at>?''',
                (scope, model_id, _timestamp(_now()))).fetchone()
            if row is None:
                return None
            artifact = self._artifact(row)
            return self._summary(row) if summary_only else artifact
        return self._guard(scope_id, read)

    def get(self, scope_id, model_id):
        return self._read(scope_id, model_id, summary_only=False)

    def get_summary(self, scope_id, model_id):
        return self._read(scope_id, model_id, summary_only=True)

    def list_for_scope(self, scope_id, *, limit=10, include_input_contract=False):
        if type(limit) is not int or not 1 <= limit <= MAX_MODELS_PER_SCOPE:
            raise ValueError('Invalid model list limit.')
        def read(connection, scope):
            rows = connection.execute('''SELECT * FROM forecast_models WHERE scope_id=? AND expires_at>?
                ORDER BY created_at DESC, model_id LIMIT ?''', (scope, _timestamp(_now()), limit)).fetchall()
            models = []
            for row in rows:
                artifact = self._artifact(row)
                summary = self._summary(row)
                if include_input_contract:
                    summary['input_contract'] = {
                        'time_column': artifact['task']['time_column'],
                        'target_column': artifact['task']['target_column'],
                        'target_unit': artifact['target_unit'],
                        'training_start': artifact['training']['start'],
                        'trained_through': artifact['training']['through'],
                        'original_issue_date': artifact['training']['issue_date'],
                        'reporting_delay_days': artifact['task'].get('reporting_delay_days', 0),
                    }
                models.append(summary)
            return models
        return self._guard(scope_id, read)

    def predict(self, scope_id, model_id, *, history=None, forecast_origin=None,
                require_new_history=False, retain_forecast=False, series_choice=None):
        if series_choice is not None and not retain_forecast:
            raise ValueError('A series can only be linked when saving a forecast.')
        model_id = _validated_job_id(model_id)
        def predict(connection, scope):
            row = connection.execute('''SELECT * FROM forecast_models
                WHERE scope_id=? AND model_id=? AND expires_at>?''',
                (scope, model_id, _timestamp(_now()))).fetchone()
            if row is None:
                raise ForecastModelNotFound('No retained model is available in this scope.')
            artifact = self._artifact(row)
            report = predict_artifact(artifact, history=history, forecast_origin=forecast_origin,
                                      require_new_history=require_new_history)
            if retain_forecast:
                prediction_id, created = store_forecast(connection, scope, report, artifact, _now())
                if prediction_id is not None:
                    forecast_series.bind_series(connection, scope, prediction_id, created=created,
                                                choice=series_choice, artifact=artifact, points=report['predictions'])
                report['retained_prediction_id'] = prediction_id
            return report
        # Keep ownership, inference and optional persistence under the deletion fence.
        return self._guard(scope_id, predict)

    def list_forecasts(self, scope_id):
        def read(connection, scope):
            return [dict(row) for row in connection.execute('''
                SELECT prediction_id, forecast_origin
                FROM forecast_outcomes WHERE scope_id=? AND expires_at>?
                ORDER BY registered_at DESC, prediction_id''', (scope, _timestamp(_now())))]
        return self._guard(scope_id, read)

    def list_series(self, scope_id):
        return self._guard(scope_id, lambda connection, scope:
                           forecast_series.list_series(connection, scope, _timestamp(_now())))

    def get_forecast(self, scope_id, prediction_id, *, actuals=None, correction=None):
        """Read a forecast or atomically apply one type of outcome update."""
        if actuals is not None and correction is not None:
            raise ValueError('Apply either actuals or a correction.')
        prediction_id = _validated_job_id(prediction_id)

        def read_or_update(connection, scope):
            now = _now()
            row = connection.execute('''SELECT * FROM forecast_outcomes
                WHERE scope_id=? AND prediction_id=? AND expires_at>?''',
                (scope, prediction_id, _timestamp(now))).fetchone()
            if row is None:
                return None
            payload = json.loads(row['forecast_json'])
            link = forecast_series.get_link(connection, scope, prediction_id)
            recorded = (forecast_series.read_series_actuals(connection, scope, link) if link is not None
                        else json.loads(row['actuals_json']))
            merged = recorded
            if actuals is not None:
                merged = merge_actuals(payload, recorded, actuals, now)
            elif correction is not None:
                count = (forecast_series.count_series_corrections(connection, scope, link['series_id'])
                         if link is not None else None)
                merged = correct_actual(payload, recorded, *correction, now=now, correction_count=count)
            view = forecast_view(row, payload, merged)
            view['series_label'] = link['label'] if link is not None else None
            if merged != recorded:
                if link is not None:
                    forecast_series.write_series_actuals(connection, scope, link['series_id'], recorded, merged)
                else:
                    connection.execute('''UPDATE forecast_outcomes SET actuals_json=?
                        WHERE scope_id=? AND prediction_id=?''',
                        (json.dumps(merged, ensure_ascii=False, allow_nan=False), scope, prediction_id))
            return view
        return self._guard(scope_id, read_or_update)

    def delete(self, scope_id, model_id):
        model_id = _validated_job_id(model_id)
        return self._guard(scope_id, lambda connection, scope: connection.execute(
            'DELETE FROM forecast_models WHERE scope_id=? AND model_id=?', (scope, model_id)).rowcount == 1)

    def purge_expired(self, scope_id):
        """Explicit scoped cleanup; reads deny expiry without extending its TTL."""
        return self._guard(scope_id, lambda connection, scope: connection.execute(
            'DELETE FROM forecast_models WHERE scope_id=? AND expires_at<=?',
            (scope, _timestamp(_now()))).rowcount)
