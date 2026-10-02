"""Create and verify a stable identity for one runtime-state directory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any


RUNTIME_STATE_IDENTITY_CONTRACT = "runtime-state-identity-v1"
IDENTITY_RELATIVE_PATH = Path("identity") / "runtime_state.json"
MAX_IDENTITY_BYTES = 4096
STATE_ID_PATTERN = re.compile(r"[0-9a-f]{64}")


class RuntimeStateError(ValueError):
    """The runtime-state identity is absent, unsafe, or malformed."""


def _state_root(value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise RuntimeStateError("Runtime state directory is invalid.")
    try:
        candidate = Path(value).expanduser().absolute()
        if candidate.is_symlink():
            raise RuntimeStateError("Runtime state directory must not be a symbolic link.")
        root = candidate.resolve()
    except RuntimeStateError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeStateError("Runtime state directory is invalid.") from error
    if not root.is_dir():
        raise RuntimeStateError("Runtime state directory is unavailable.")
    return root


def _state_id(value: object) -> str:
    if not isinstance(value, str) or STATE_ID_PATTERN.fullmatch(value) is None:
        raise RuntimeStateError(
            "Runtime state identity must be a 64-character lowercase hexadecimal value."
        )
    return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeStateError("Runtime state identity contains duplicate fields.")
        result[key] = value
    return result


def _validate_document(document: object) -> dict[str, str]:
    if (
        not isinstance(document, dict)
        or set(document) != {"contract", "state_id", "created_at"}
        or document.get("contract") != RUNTIME_STATE_IDENTITY_CONTRACT
    ):
        raise RuntimeStateError("Runtime state identity contract is invalid.")
    state_id = _state_id(document.get("state_id"))
    created_at = document.get("created_at")
    try:
        parsed = datetime.fromisoformat(created_at)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError()
    except (TypeError, ValueError) as error:
        raise RuntimeStateError("Runtime state identity timestamp is invalid.") from error
    return {
        "contract": RUNTIME_STATE_IDENTITY_CONTRACT,
        "state_id": state_id,
        "created_at": created_at,
    }


def _read_identity(root: Path) -> dict[str, str]:
    path = root / IDENTITY_RELATIVE_PATH
    try:
        initial = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeStateError("Runtime state identity is missing.") from error
    except OSError as error:
        raise RuntimeStateError("Runtime state identity is unavailable.") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
        raise RuntimeStateError("Runtime state identity must be a regular file.")
    if initial.st_size > MAX_IDENTITY_BYTES:
        raise RuntimeStateError("Runtime state identity exceeds its size limit.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeStateError("Runtime state identity is unavailable.") from error
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != initial.st_dev
            or observed.st_ino != initial.st_ino
        ):
            raise RuntimeStateError("Runtime state identity changed while being read.")
        with os.fdopen(descriptor, "rb") as reader:
            encoded = reader.read(MAX_IDENTITY_BYTES + 1)
            descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(encoded) > MAX_IDENTITY_BYTES:
        raise RuntimeStateError("Runtime state identity exceeds its size limit.")
    try:
        document = json.loads(encoded, object_pairs_hook=_unique_keys)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise RuntimeStateError("Runtime state identity is malformed.") from error
    return _validate_document(document)


def _fingerprint(state_id: str) -> str:
    return hashlib.sha256(state_id.encode("ascii")).hexdigest()[:12]


def initialize_runtime_state(state_dir: object) -> dict[str, str]:
    """Create the identity exactly once, or return the existing valid identity."""

    root = _state_root(state_dir)
    identity_directory = root / IDENTITY_RELATIVE_PATH.parent
    try:
        identity_directory.mkdir(mode=0o700, exist_ok=True)
        directory_info = identity_directory.lstat()
    except OSError as error:
        raise RuntimeStateError("Runtime state identity directory is unavailable.") from error
    if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
        raise RuntimeStateError("Runtime state identity directory must be a regular directory.")

    path = root / IDENTITY_RELATIVE_PATH
    if path.exists() or path.is_symlink():
        return _read_identity(root)

    document = {
        "contract": RUNTIME_STATE_IDENTITY_CONTRACT,
        "state_id": secrets.token_hex(32),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    encoded = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return _read_identity(root)
    except OSError as error:
        raise RuntimeStateError("Runtime state identity could not be created.") from error
    try:
        with os.fdopen(descriptor, "wb") as writer:
            writer.write(encoded)
            writer.flush()
            os.fsync(writer.fileno())
        directory_descriptor = os.open(identity_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise RuntimeStateError("Runtime state identity could not be persisted.") from error
    return _read_identity(root)


def inspect_runtime_state(state_dir: object, expected_state_id: object) -> dict[str, Any]:
    """Return a secret-free continuity result for readiness and operations."""

    try:
        expected = _state_id(expected_state_id)
    except RuntimeStateError:
        return {
            "contract": RUNTIME_STATE_IDENTITY_CONTRACT,
            "status": "invalid_expected_identity",
            "verified": False,
            "fingerprint": None,
        }
    try:
        document = _read_identity(_state_root(state_dir))
    except RuntimeStateError:
        return {
            "contract": RUNTIME_STATE_IDENTITY_CONTRACT,
            "status": "missing_or_invalid_identity",
            "verified": False,
            "fingerprint": None,
        }
    matches = hmac.compare_digest(document["state_id"], expected)
    return {
        "contract": RUNTIME_STATE_IDENTITY_CONTRACT,
        "status": "verified" if matches else "identity_mismatch",
        "verified": matches,
        "fingerprint": _fingerprint(document["state_id"]),
    }
