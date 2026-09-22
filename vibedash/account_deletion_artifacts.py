"""Safe cleanup of VibeDash artifacts owned by one analysis scope.

This module deliberately does not use paths stored in payloads.  It scans the
configured artifact directories, accepts only the server-generated filename
patterns, and proves ownership from the persisted session owner field or a
job row whose ``scope_id`` matches the requested scope.

The caller is expected to hold the analysis-jobs SQLite write lock and to have
already confirmed that the scope has no active jobs.  A malformed or
unreadable exact-pattern session/job record causes a generic retryable error:
ownership cannot safely be established, so no artifact deletion is attempted.
"""

from __future__ import annotations

import json
import errno
import os
import re
import stat
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .exporter import (
    EXPORT_FILE_PATTERN,
    SESSION_FILE_PATTERN,
    UPLOAD_FILE_PATTERN,
    _resolve_exports_dir,
    _resolve_sessions_dir,
)


_SCOPE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_EXPORT_SESSION_ID_PATTERN = re.compile(
    r"^vibedash_export_(?P<session_id>[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})_[0-9]{8}_[0-9]{6}\.html$"
)


class AccountArtifactCleanupError(RuntimeError):
    """A proven artifact could not be removed safely.

    The message intentionally contains no path, filename, or persisted user
    data.  The account-deletion route can return a generic retryable failure
    without leaking artifact metadata.
    """


def _validated_scope_id(scope_id: str) -> str:
    if not isinstance(scope_id, str) or not _SCOPE_ID_PATTERN.fullmatch(scope_id):
        raise ValueError("Invalid analysis scope identifier.")
    return scope_id


@contextmanager
def _safe_directory(directory: Path):
    """Yield an open directory descriptor without following its final link.

    Keeping the descriptor open for the complete scan/removal pass avoids a
    pathname TOCTOU: checking ``directory`` with ``lstat`` and then calling
    ``iterdir`` would allow a writer to swap that path for a symlink between
    the two operations and make later unlink calls target another tree.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(directory, flags)
    except FileNotFoundError:
        yield None
        return
    except OSError as error:
        if getattr(error, "errno", None) in (errno.ENOTDIR, errno.ELOOP):
            yield None
            return
        raise AccountArtifactCleanupError(
            "Account artifact cleanup could not inspect storage."
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            yield None
            return
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _directory_names(descriptor: int | None) -> list[str]:
    if descriptor is None:
        return []
    try:
        return list(os.listdir(descriptor))
    except OSError as error:
        raise AccountArtifactCleanupError(
            "Account artifact cleanup could not inspect storage."
        ) from error


def _regular_file(path: Path, *, directory_fd: int | None = None, name: str | None = None) -> bool:
    """Check a candidate with lstat so a symlink is never followed."""
    try:
        if directory_fd is not None and name is not None:
            mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
        else:
            mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return False
    except OSError as error:
        raise AccountArtifactCleanupError(
            "Account artifact cleanup could not inspect storage."
        ) from error
    return stat.S_ISREG(mode)


def _safe_unlink(
    path: Path,
    *,
    directory_fd: int | None = None,
    name: str | None = None,
) -> bool:
    """Remove one proven regular file, translating deletion failures safely."""
    if not _regular_file(path, directory_fd=directory_fd, name=name):
        return False
    try:
        # unlink never follows a symlink.  The lstat immediately above also
        # keeps symlink candidates out of the deletion set.
        if directory_fd is not None and name is not None:
            os.unlink(name, dir_fd=directory_fd)
        else:
            path.unlink()
    except FileNotFoundError:
        # A concurrent retention pass may have removed it.  The desired end
        # state is already true, so this remains an idempotent success.
        return False
    except OSError as error:
        raise AccountArtifactCleanupError(
            "Account artifact cleanup could not remove a proven artifact."
        ) from error
    return True


def _valid_upload_name(value: Any) -> bool:
    return isinstance(value, str) and UPLOAD_FILE_PATTERN.fullmatch(value) is not None


def _session_upload_names(data: Mapping[str, Any]) -> set[str]:
    value = data.get("stored_filename")
    return {value} if _valid_upload_name(value) else set()


def _job_upload_names(payload: Any) -> set[str]:
    """Extract only dashboard and period-comparison generated references."""
    if not isinstance(payload, Mapping):
        return set()
    names: set[str] = set()
    stored_filename = payload.get("stored_filename")
    if _valid_upload_name(stored_filename):
        names.add(stored_filename)
    for key in ("baseline", "current"):
        item = payload.get(key)
        if isinstance(item, Mapping):
            stored_filename = item.get("stored_filename")
            if _valid_upload_name(stored_filename):
                names.add(stored_filename)
    return names


def _canonical_session_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        normalized = str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError):
        return None
    return normalized if normalized == value else None


def _session_id_from_filename(name: str) -> str | None:
    if SESSION_FILE_PATTERN.fullmatch(name) is None:
        return None
    return _canonical_session_id(name[:-5])


def _read_session(
    path: Path,
    *,
    directory_fd: int | None = None,
    name: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Read one candidate; the bool reports unknown/unreadable metadata."""
    # Do not split an ownership check from a pathname open.  A final symlink
    # can be swapped in between lstat/is_file and open(), causing cleanup to
    # parse metadata from outside the configured sessions directory.  Opening
    # with O_NOFOLLOW and checking the descriptor's type makes the record
    # identity part of the read operation itself.
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        if directory_fd is not None and name is not None:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        else:
            descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None, False
    except OSError as error:
        if getattr(error, "errno", None) == errno.ELOOP:
            return None, False
        raise AccountArtifactCleanupError(
            "Account artifact cleanup could not inspect storage."
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None, False
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = None
            data = json.load(handle)
    except (OSError, UnicodeError, ValueError, TypeError):
        return None, True
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(data, dict):
        return None, True
    return data, False


def _iter_jobs(
    jobs: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[Any]:
    if jobs is None:
        return []
    if isinstance(jobs, Mapping):
        return [jobs]
    try:
        return list(jobs)
    except (TypeError, ValueError):
        return [None]


def purge_account_artifacts(
    scope_id: str,
    jobs: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    upload_dir: str | os.PathLike[str],
    sessions_dir: str | os.PathLike[str] | None = None,
    exports_dir: str | os.PathLike[str] | None = None,
    protected_upload_names: Iterable[str] | None = None,
) -> dict[str, int]:
    """Delete regular VibeDash artifacts proven to belong to ``scope_id``.

    ``jobs`` must be rows from :class:`AnalysisJobStore` (or an already
    filtered equivalent) and each row must carry its ``scope_id``.  Rows with
    a missing scope or corrupt payload cause a generic retryable error.  Known
    foreign session/job references protect an upload from deletion; callers
    may pass additional known foreign references through
    ``protected_upload_names``.

    Counts use the existing exporter convention: ``removed_sessions``,
    ``removed_uploads``, and ``removed_exports``.  Missing directories and
    symlink/non-regular candidates are intentionally no-ops.
    """
    normalized_scope_id = _validated_scope_id(scope_id)
    sessions_path = (
        Path(sessions_dir)
        if sessions_dir is not None
        else Path(_resolve_sessions_dir(None))
    )
    exports_path = (
        Path(exports_dir)
        if exports_dir is not None
        else Path(_resolve_exports_dir(None))
    )
    uploads_path = Path(upload_dir)

    owned_session_ids: set[str] = set()
    session_owners: dict[str, str] = {}
    owned_upload_names: set[str] = set()
    foreign_upload_names: set[str] = set()
    unknown_metadata = False
    session_candidates: list[tuple[str, dict[str, Any]]] = []
    # Keep a duplicate of the validated directory descriptor so the unlink
    # pass cannot be redirected by replacing ``sessions_path`` after the
    # metadata scan.  Reopening the pathname would reintroduce a directory
    # TOCTOU even though each individual open uses O_NOFOLLOW.
    sessions_delete_fd: int | None = None

    with _safe_directory(sessions_path) as sessions_fd:
        for name in _directory_names(sessions_fd):
            session_id = _session_id_from_filename(name)
            if session_id is None:
                continue
            candidate = sessions_path / name
            data, unknown = _read_session(
                candidate,
                directory_fd=sessions_fd,
                name=name,
            )
            if unknown:
                unknown_metadata = True
                continue
            if data is None:
                continue
            owner = data.get("analysis_scope_id")
            if not isinstance(owner, str) or _SCOPE_ID_PATTERN.fullmatch(owner) is None:
                unknown_metadata = True
                continue
            session_candidates.append((name, data))
            session_owners[session_id] = owner
            if owner == normalized_scope_id:
                owned_session_ids.add(session_id)
                owned_upload_names.update(_session_upload_names(data))
            else:
                foreign_upload_names.update(_session_upload_names(data))

        if sessions_fd is not None:
            try:
                sessions_delete_fd = os.dup(sessions_fd)
            except OSError as error:
                raise AccountArtifactCleanupError(
                    "Account artifact cleanup could not inspect storage."
                ) from error

    owned_job_session_ids: set[str] = set()
    protected_session_ids: set[str] = set(
        value
        for value in getattr(jobs, "foreign_scope_session_ids", ())
        if _canonical_session_id(value) is not None
    )
    for job in _iter_jobs(jobs):
        if not isinstance(job, Mapping):
            unknown_metadata = True
            continue
        row_scope = job.get("scope_id", job.get("analysis_scope_id"))
        if not isinstance(row_scope, str):
            # An unscoped row cannot prove either ownership or foreignness.
            unknown_metadata = True
            foreign_upload_names.update(_job_upload_names(job.get("payload")))
            continue
        payload = job.get("payload")
        if not isinstance(payload, Mapping):
            unknown_metadata = True
            payload_names: set[str] = set()
        else:
            payload_names = _job_upload_names(payload)
        if row_scope == normalized_scope_id:
            owned_upload_names.update(payload_names)
            raw_session_id = job.get("session_id")
            job_session_id = _canonical_session_id(raw_session_id)
            if raw_session_id is not None and job_session_id is None:
                # A non-empty malformed session reference is corrupt metadata,
                # not evidence for an export.  Fail closed rather than risk
                # interpreting a path-shaped value elsewhere.
                unknown_metadata = True
            elif job_session_id is not None:
                owner = session_owners.get(job_session_id)
                if owner is None:
                    # The session may have already expired independently.  In
                    # that case its export cannot be proven owned and is left
                    # for normal retention cleanup.
                    pass
                elif owner == normalized_scope_id:
                    owned_job_session_ids.add(job_session_id)
                else:
                    # A durable job and a session file disagree about
                    # ownership.  Never delete the export on either claim.
                    unknown_metadata = True
        else:
            foreign_upload_names.update(payload_names)
            foreign_session_id = _canonical_session_id(job.get("session_id"))
            if job.get("session_id") is not None and foreign_session_id is None:
                unknown_metadata = True
            elif foreign_session_id is not None:
                protected_session_ids.add(foreign_session_id)

    for name in protected_upload_names or ():
        if _valid_upload_name(name):
            foreign_upload_names.add(name)

    if unknown_metadata:
        if sessions_delete_fd is not None:
            os.close(sessions_delete_fd)
            sessions_delete_fd = None
        raise AccountArtifactCleanupError(
            "Account artifact cleanup could not establish artifact ownership."
        )

    # Keep session metadata until every dependent upload/export removal has
    # succeeded.  File deletion is not transactional: if an upload or export
    # unlink fails after the session row is removed, a retry would no longer
    # be able to prove ownership of a session-only export.  Removing the
    # metadata last makes the operation safely retryable and prevents those
    # exports from becoming permanently unidentifiable.
    removed_sessions = 0
    removed_uploads = 0
    removed_exports = 0
    try:
        protected_names = foreign_upload_names
        with _safe_directory(uploads_path) as uploads_fd:
            upload_candidates = {
                name for name in _directory_names(uploads_fd)
                if UPLOAD_FILE_PATTERN.fullmatch(name) is not None
            }
            for name in sorted(owned_upload_names - protected_names):
                if name not in upload_candidates:
                    continue
                if _safe_unlink(
                    uploads_path / name,
                    directory_fd=uploads_fd,
                    name=name,
                ):
                    removed_uploads += 1

        export_session_ids = (
            owned_session_ids | owned_job_session_ids
        ) - protected_session_ids
        with _safe_directory(exports_path) as exports_fd:
            for name in _directory_names(exports_fd):
                if EXPORT_FILE_PATTERN.fullmatch(name) is None:
                    continue
                match = _EXPORT_SESSION_ID_PATTERN.fullmatch(name)
                if match is None or match.group("session_id") not in export_session_ids:
                    continue
                if _safe_unlink(
                    exports_path / name,
                    directory_fd=exports_fd,
                    name=name,
                ):
                    removed_exports += 1

        for name, _data in session_candidates:
            session_id = _session_id_from_filename(name)
            if session_id not in owned_session_ids:
                continue
            if _safe_unlink(
                sessions_path / name,
                directory_fd=sessions_delete_fd,
                name=name,
            ):
                removed_sessions += 1
    finally:
        if sessions_delete_fd is not None:
            os.close(sessions_delete_fd)

    return {
        "removed_sessions": removed_sessions,
        "removed_uploads": removed_uploads,
        "removed_exports": removed_exports,
    }


__all__ = [
    "AccountArtifactCleanupError",
    "purge_account_artifacts",
]
