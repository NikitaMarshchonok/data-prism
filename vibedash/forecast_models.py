"""Owner-scoped model storage preparation; deliberately not wired to HTTP.

Uses the existing analysis database and deletion fence. Persistent infrastructure
is operator-declared and mount continuity checked, not physically proven here.
No imported Python objects, automatic web saving, activation or retraining.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from src.forecast_artifact import MAX_ARTIFACT_BYTES, predict_artifact, validate_artifact
from src.runtime_storage import evaluate_runtime_storage_contract
from .analysis_jobs import AnalysisJobStore, _validated_job_id, _validated_scope_id


CONTRACT = 'scoped-forecast-model-store-v1'
MAX_MODELS_PER_SCOPE = 10
MAX_MODELS_TOTAL = 100
MAX_BYTES_PER_SCOPE = 16 * 1024 * 1024
MAX_BYTES_TOTAL = 128 * 1024 * 1024


class ForecastModelStorageUnavailable(ValueError):
    """The persistent-state declaration or continuity check is not satisfied."""


class ForecastModelCapacityError(ValueError):
    """Explicit capacity is exhausted; no retained model is silently removed."""


def _now():
    return datetime.now(timezone.utc)


def _timestamp(value):
    return value.isoformat(timespec='microseconds')


def _serialized(artifact):
    validate_artifact(artifact)
    return json.dumps(artifact, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


class ForecastModelStore:
    """Internal server API: scope must come from authenticated server context.

    A scope string alone is not authentication. There is intentionally no web
    route, guest fallback, account migration or customer opt-in in this slice.
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
        if type(retention_days) is not int or not 1 <= retention_days <= 30:
            raise ValueError('Choose an explicit retention of 1–30 days.')
        serialized = _serialized(artifact)
        artifact = json.loads(serialized)  # immutable copy of the validated input
        size = len(serialized.encode('utf-8'))
        if size > MAX_ARTIFACT_BYTES:
            raise ValueError('Model exceeds the artifact byte budget.')
        def save(connection, scope):
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
                                size, _timestamp(now), _timestamp(now + timedelta(days=retention_days))))
            saved = connection.execute('SELECT * FROM forecast_models WHERE scope_id=? AND model_id=?',
                                       (scope, artifact['model_id'])).fetchone()
            return self._summary(saved) | {'created': True}
        return self._guard(scope_id, save)

    @staticmethod
    def _summary(row):
        return {'contract': CONTRACT, 'model_id': row['model_id'], 'model_sha256': row['artifact_sha256'],
                'created_at': row['created_at'], 'expires_at': row['expires_at'],
                'artifact_bytes': row['artifact_bytes'], 'production_readiness': 'not_established'}

    def get(self, scope_id, model_id):
        model_id = _validated_job_id(model_id)
        def read(connection, scope):
            row = connection.execute('''SELECT * FROM forecast_models
                WHERE scope_id=? AND model_id=? AND expires_at>?''',
                (scope, model_id, _timestamp(_now()))).fetchone()
            return self._artifact(row) if row is not None else None
        return self._guard(scope_id, read)

    def list_for_scope(self, scope_id, *, limit=10):
        if type(limit) is not int or not 1 <= limit <= MAX_MODELS_PER_SCOPE:
            raise ValueError('Invalid model list limit.')
        def read(connection, scope):
            rows = connection.execute('''SELECT * FROM forecast_models WHERE scope_id=? AND expires_at>?
                ORDER BY created_at DESC, model_id LIMIT ?''', (scope, _timestamp(_now()), limit)).fetchall()
            for row in rows:
                self._artifact(row)
            return [self._summary(row) for row in rows]
        return self._guard(scope_id, read)

    def predict(self, scope_id, model_id, *, history=None, forecast_origin=None):
        model_id = _validated_job_id(model_id)
        def predict(connection, scope):
            row = connection.execute('''SELECT * FROM forecast_models
                WHERE scope_id=? AND model_id=? AND expires_at>?''',
                (scope, model_id, _timestamp(_now()))).fetchone()
            if row is None:
                raise ValueError('No retained model is available in this scope.')
            return predict_artifact(self._artifact(row), history=history, forecast_origin=forecast_origin)
        # Ownership and the deletion fence stay held through frozen inference.
        # This does not persist the returned predictions or match actuals.
        return self._guard(scope_id, predict)

    def delete(self, scope_id, model_id):
        model_id = _validated_job_id(model_id)
        return self._guard(scope_id, lambda connection, scope: connection.execute(
            'DELETE FROM forecast_models WHERE scope_id=? AND model_id=?', (scope, model_id)).rowcount == 1)

    def purge_expired(self, scope_id):
        """Explicit scoped cleanup; reads deny expiry without extending its TTL."""
        return self._guard(scope_id, lambda connection, scope: connection.execute(
            'DELETE FROM forecast_models WHERE scope_id=? AND expires_at<=?',
            (scope, _timestamp(_now()))).rowcount)
