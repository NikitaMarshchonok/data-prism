"""Offline, local runtime snapshots. Never replaces an existing directory."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import time


CONTRACT = 'data-prism-runtime-backup-v1'
STATE_ROOTS = frozenset({'uploads', 'reports', 'baselines', 'jobs', 'drift', 'sessions', 'exports'})
DATABASES = {
    'jobs/analysis_jobs.sqlite3': {'analysis_jobs'},
    'drift/drift_history.sqlite3': {'drift_runs', 'drift_alerts'},
}
MAX_FILES = 20000
MAX_BYTES = 2 * 1024 ** 3
MAX_MANIFEST_BYTES = 8 * 1024 ** 2
MAX_DEPTH = 8
SQLITE_TIMEOUT_SECONDS = 30


class BackupError(ValueError):
    """A safe precondition or integrity check failed; messages contain no data."""


def _path(value):
    try:
        candidate = Path(value).expanduser().absolute()
        if candidate.is_symlink():
            raise BackupError('The selected directory must not be a symbolic link.')
        path = candidate.resolve()
    except BackupError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        # Path parsing/resolution errors are operator input failures.  Do not
        # echo the supplied path (it may contain secrets or control bytes).
        raise BackupError('The selected path is invalid or cannot be resolved.') from error
    if path == Path(path.anchor) or path == Path.home().resolve():
        raise BackupError('Select a dedicated runtime or backup directory, not a broad root.')
    return path


def _separate(source, destination):
    if source == destination or source in destination.parents or destination in source.parents:
        raise BackupError('Source and destination must be separate, non-nested directories.')
    if destination.exists() or destination.is_symlink():
        raise BackupError('Destination already exists. Select a new directory; nothing is overwritten.')
    if not destination.parent.is_dir():
        raise BackupError('Create the destination parent directory first.')


def _relative(name):
    if not isinstance(name, str) or len(name) > 1024 or '\\' in name:
        raise BackupError('Unsupported path in runtime snapshot.')
    parts = name.split('/')
    if (not parts or parts[0] not in STATE_ROOTS or len(parts) > MAX_DEPTH
            or any(not part or part.startswith('.') or ':' in part or '\x00' in part for part in parts)
            or PurePosixPath(name).is_absolute()):
        raise BackupError('Unsupported path in runtime snapshot.')
    return name


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _directory_entries(directory):
    """Yield a bounded, explicitly closable directory iterator."""
    try:
        entries = os.scandir(directory)
    except OSError as error:
        raise BackupError('Runtime directory is unavailable.') from error
    try:
        yield entries
    finally:
        entries.close()


def _inventory(root, *, live=False):
    if not root.is_dir() or root.is_symlink():
        raise BackupError('Runtime payload directory is missing or is a symbolic link.')
    result, folded, total, entries = {}, set(), 0, 0

    def walk(directory):
        nonlocal total, entries
        with _directory_entries(directory) as directory_entries:
            for entry in directory_entries:
                entries += 1
                if entries > MAX_FILES * 2:
                    raise BackupError('Runtime directory exceeds the entry limit.')
                path = Path(entry.path)
                name = _relative(path.relative_to(root).as_posix())
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise BackupError('Symbolic links are not supported in runtime snapshots.')
                if stat.S_ISDIR(info.st_mode):
                    if name.casefold() in folded:
                        raise BackupError('Case-insensitive path collision in runtime snapshot.')
                    folded.add(name.casefold())
                    walk(path)
                    continue
                if '/' not in name:
                    raise BackupError('Runtime top-level entries must be dedicated directories.')
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise BackupError('Only regular, non-hardlinked files are supported.')
                sidecar_db = next((db for db in DATABASES if name in {db + '-wal', db + '-shm'}), None)
                if sidecar_db:
                    if not live:
                        raise BackupError('Unexpected SQLite sidecar in snapshot payload.')
                    try:
                        database_info = (root / sidecar_db).lstat()
                    except OSError as error:
                        raise BackupError('SQLite sidecar has no corresponding database.') from error
                    if (not stat.S_ISREG(database_info.st_mode)
                            or database_info.st_nlink != 1):
                        raise BackupError('SQLite sidecar has no corresponding regular database.')
                    continue  # SQLite backup API includes committed WAL pages.
                if name.casefold() in folded:
                    raise BackupError('Case-insensitive path collision in runtime snapshot.')
                folded.add(name.casefold())
                total += info.st_size
                if len(result) >= MAX_FILES or total > MAX_BYTES:
                    raise BackupError('Runtime snapshot exceeds file-count or size limits.')
                result[name] = _signature(info)

    walk(root)
    if 'jobs/analysis_jobs.sqlite3' not in result:
        raise BackupError('Expected a dedicated DATA_PRISM_STATE_DIR with an analysis-job database.')
    return result


def _wal_signature(root):
    result = {}
    for name in DATABASES:
        wal = root / (name + '-wal')
        if wal.exists() and wal.lstat().st_size:
            result[name] = _signature(wal.lstat())
    return result


@contextmanager
def _regular_file(path):
    # lstat must precede open: opening a FIFO without O_NONBLOCK can hang
    # verification indefinitely.  The descriptor is checked again to cover a
    # replacement between lstat and open (operator-controlled parent races
    # are outside this offline contract).
    try:
        initial = path.lstat()
    except (OSError, ValueError) as error:
        raise BackupError('Runtime snapshot file is unavailable.') from error
    if stat.S_ISLNK(initial.st_mode):
        raise BackupError('Symbolic links are not supported in runtime snapshots.')
    if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
        raise BackupError('Only regular, non-hardlinked files are supported.')
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError) as error:
        raise BackupError('Runtime snapshot file is unavailable.') from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise BackupError('Only regular, non-hardlinked files are supported.')
        source = os.fdopen(descriptor, 'rb')
    except BackupError:
        os.close(descriptor)
        raise
    except (OSError, ValueError) as error:
        os.close(descriptor)
        raise BackupError('Runtime snapshot file is unavailable.') from error
    with source:
        yield source


def _file_record(path):
    digest, size = hashlib.sha256(), 0
    with _regular_file(path) as source:
        initial = os.fstat(source.fileno())
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise BackupError('Snapshot file exceeds size limit.')
            digest.update(chunk)
        if _signature(initial) != _signature(os.fstat(source.fileno())):
            raise BackupError('File changed during verification. Stop all writers and retry.')
    if not 0 <= initial.st_mtime_ns <= 2**63 - 1:
        raise BackupError('Unsupported file modification time.')
    return {'size': size, 'sha256': digest.hexdigest(), 'mtime_ns': initial.st_mtime_ns}


def _copy_file(source, destination, expected=None):
    _private_parents(destination.parent)
    digest, size = hashlib.sha256(), 0
    with _regular_file(source) as reader:
        info = os.fstat(reader.fileno())
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as writer:
            while chunk := reader.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise BackupError('Snapshot file exceeds size limit.')
                writer.write(chunk)
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if _signature(info) != _signature(os.fstat(reader.fileno())):
            raise BackupError('File changed during copy. Stop all writers and retry.')
    if expected and (size != expected['size'] or digest.hexdigest() != expected['sha256']):
        raise BackupError('File integrity check failed during restore.')
    modified = expected['mtime_ns'] if expected else info.st_mtime_ns
    os.utime(destination, ns=(modified, modified))


def _private_parents(path):
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    if path.is_symlink() or not path.is_dir():
        raise BackupError('Destination parent must be a regular directory.')
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)


@contextmanager
def _database(path):
    if path.is_symlink() or not path.is_file():
        raise BackupError('Database is missing or is a symbolic link.')
    with closing(sqlite3.connect(
            path.as_uri() + '?mode=ro', uri=True,
            timeout=min(5, max(0, SQLITE_TIMEOUT_SECONDS)))) as connection:
        connection.execute('PRAGMA query_only = ON')
        connection.execute('PRAGMA trusted_schema = OFF')
        deadline = time.monotonic() + SQLITE_TIMEOUT_SECONDS
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        yield connection


def _check_database(path, name):
    with _database(path) as connection:
        if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise BackupError('SQLite integrity verification failed.')
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if not DATABASES[name].issubset(tables):
            raise BackupError('Database does not have the expected runtime schema.')
        if name == 'jobs/analysis_jobs.sqlite3':
            active = connection.execute("SELECT COUNT(*) FROM analysis_jobs WHERE status IN ('queued', 'running')").fetchone()[0]
            if active:
                raise BackupError('Queued or running jobs remain. Drain work before taking an offline snapshot.')


def _snapshot_database(source, destination):
    _private_parents(destination.parent)
    destination.touch(mode=0o600, exist_ok=False)
    deadline = time.monotonic() + SQLITE_TIMEOUT_SECONDS

    def progress(_status, _remaining, total):
        if time.monotonic() > deadline or total * page_size > MAX_BYTES:
            raise BackupError('SQLite snapshot exceeded its time or size limit.')

    with _database(source) as reader, closing(sqlite3.connect(destination)) as writer:
        page_size = reader.execute('PRAGMA page_size').fetchone()[0]
        reader.backup(writer, pages=128, progress=progress, sleep=0.05)
        writer.execute('PRAGMA journal_mode = DELETE')
    modified = source.stat().st_mtime_ns
    os.utime(destination, ns=(modified, modified))


def _offline(acknowledged):
    if acknowledged is not True:
        raise BackupError('Stop the web service and all other writers, then explicitly confirm offline mode.')


def create_backup(state_dir, output_dir, *, offline=False, revision='unknown'):
    _offline(offline)
    if not isinstance(revision, str) or (revision != 'unknown' and not re.fullmatch(r'[0-9a-f]{7,40}', revision)):
        raise BackupError('Revision must be a Git commit hash or unknown.')
    source, output = _path(state_dir), _path(output_dir)
    _separate(source, output)
    initial = _inventory(source, live=True)
    initial_wal = _wal_signature(source)
    for name in DATABASES.keys() & initial.keys():
        _check_database(source / name, name)

    # An exclusively created private directory; errors leave an incomplete copy,
    # never delete or replace user data. manifest.json is the completion marker.
    output.mkdir(mode=0o700)
    payload = output / 'state'
    payload.mkdir(mode=0o700)
    for name in initial:
        if name in DATABASES:
            _snapshot_database(source / name, payload / name)
        else:
            _copy_file(source / name, payload / name)
    if _inventory(source, live=True) != initial or _wal_signature(source) != initial_wal:
        raise BackupError('Runtime state changed during backup. Stop all writers and retry.')
    stored = _inventory(payload)
    for name in DATABASES.keys() & stored.keys():
        _check_database(payload / name, name)
    records = {name: _file_record(payload / name) for name in stored}
    manifest = {
        'contract': CONTRACT, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'revision': revision, 'files': records,
    }
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode('utf-8')
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise BackupError('Manifest exceeds size limit.')
    descriptor = os.open(output / '.manifest-incomplete', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as writer:
        writer.write(encoded)
        writer.flush()
        os.fsync(writer.fileno())
    os.rename(output / '.manifest-incomplete', output / 'manifest.json')
    return _summary(manifest)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupError('Duplicate manifest key.')
        result[key] = value
    return result


def _verified_manifest(backup_dir):
    backup = _path(backup_dir)
    if not backup.is_dir():
        raise BackupError('Backup is incomplete or contains unexpected entries.')
    top_level = []
    with _directory_entries(backup) as entries:
        for index, entry in enumerate(entries):
            if index >= 2:
                raise BackupError('Backup is incomplete or contains unexpected entries.')
            top_level.append(entry.name)
    if set(top_level) != {'state', 'manifest.json'}:
        raise BackupError('Backup is incomplete or contains unexpected entries.')
    with _regular_file(backup / 'manifest.json') as reader:
        encoded = reader.read(MAX_MANIFEST_BYTES + 1)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise BackupError('Manifest exceeds size limit.')
    try:
        manifest = json.loads(encoded, object_pairs_hook=_unique_keys)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise BackupError('Invalid backup manifest.') from error
    if (not isinstance(manifest, dict) or set(manifest) != {'contract', 'created_at', 'revision', 'files'}
            or manifest['contract'] != CONTRACT or not isinstance(manifest['files'], dict)):
        raise BackupError('Unsupported backup contract.')
    if (not isinstance(manifest['revision'], str)
            or not re.fullmatch(r'(?:unknown|[0-9a-f]{7,40})', manifest['revision'])):
        raise BackupError('Invalid backup revision.')
    try:
        if datetime.fromisoformat(manifest['created_at']).tzinfo is None:
            raise ValueError()
    except (TypeError, ValueError) as error:
        raise BackupError('Invalid backup timestamp.') from error
    files, total = manifest['files'], 0
    if len(files) > MAX_FILES:
        raise BackupError('Manifest exceeds file-count limit.')
    for name, record in files.items():
        _relative(name)
        if (not isinstance(record, dict) or set(record) != {'size', 'sha256', 'mtime_ns'}
                or type(record['size']) is not int or not 0 <= record['size'] <= MAX_BYTES
                or type(record['mtime_ns']) is not int or not 0 <= record['mtime_ns'] <= 2**63 - 1
                or not isinstance(record['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
            raise BackupError('Invalid file metadata in backup manifest.')
        total += record['size']
    if total > MAX_BYTES:
        raise BackupError('Manifest exceeds total size limit.')
    actual = _inventory(backup / 'state')
    if actual.keys() != files.keys():
        raise BackupError('Backup file inventory does not match its manifest.')
    for name, expected in files.items():
        observed = _file_record(backup / 'state' / name)
        if observed['size'] != expected['size'] or observed['sha256'] != expected['sha256']:
            raise BackupError('Backup checksum verification failed.')
    for name in DATABASES.keys() & files.keys():
        _check_database(backup / 'state' / name, name)
    return backup, manifest


def _summary(manifest):
    return {
        'contract': CONTRACT, 'created_at': manifest['created_at'], 'revision': manifest['revision'],
        'file_count': len(manifest['files']),
        'total_bytes': sum(item['size'] for item in manifest['files'].values()),
    }


def verify_backup(backup_dir):
    _, manifest = _verified_manifest(backup_dir)
    return _summary(manifest)


def restore_backup(backup_dir, destination_dir, *, offline=False):
    _offline(offline)
    source, destination = _path(backup_dir), _path(destination_dir)
    _separate(source, destination)
    _, manifest = _verified_manifest(source)
    destination.mkdir(mode=0o700)  # Existing targets, including empty ones, fail.
    for name, expected in manifest['files'].items():
        _copy_file(source / 'state' / name, destination / name, expected)
    for name in DATABASES.keys() & manifest['files'].keys():
        _check_database(destination / name, name)
    return _summary(manifest)
