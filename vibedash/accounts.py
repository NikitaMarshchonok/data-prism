"""The account persistence boundary used by VibeDash.

The store owns a small, dedicated SQLite database. It deliberately supports
one schema version only: an empty database is created transactionally and an
exact v1 database can be reopened, while anything else fails closed.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from werkzeug.security import check_password_hash, generate_password_hash


ACCOUNT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+$")
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024
MAX_EMAIL_LENGTH = 254
DEFAULT_MAX_FAILURES = 5
DEFAULT_THROTTLE_WINDOW_SECONDS = 15 * 60
DEFAULT_LOCKOUT_SECONDS = 15 * 60
DEFAULT_BUSY_TIMEOUT_MS = 5_000
MAX_THROTTLE_ROWS = 10_000
# These bounds preserve Werkzeug's generated defaults while ensuring that a
# corrupted database cannot turn an authentication check into an unbounded
# CPU/RAM allocation.  Werkzeug's scrypt implementation sets maxmem to
# 132 * n * r * p, so the product bound is the relevant memory contract.
MAX_SCRYPT_MEMORY_BYTES = 64 * 1024 * 1024
MAX_PBKDF2_ITERATIONS = 2_000_000
SCOPE_HMAC_DOMAIN = b"vibedash-account-scope-v1\x00"
CREDENTIAL_HMAC_DOMAIN = b"vibedash-account-credential-v1\x00"
CREDENTIAL_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SESSION_CREDENTIAL_KEY = "_vibedash_session_credential"
SCHEMA_VERSION = 1


class AccountValidationError(ValueError):
    """A safe, deterministic input validation failure."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        self.code = code
        self.field = field
        super().__init__(message)


class AccountStoreSchemaError(RuntimeError):
    """The dedicated accounts database is absent, partial, or incompatible."""


def normalize_email(email: str) -> str:
    """Return the canonical email representation used for uniqueness."""
    if not isinstance(email, str):
        raise AccountValidationError("email_type", "Email must be a string.", field="email")
    if len(email) > MAX_EMAIL_LENGTH:
        raise AccountValidationError(
            "email_too_long", f"Email must be at most {MAX_EMAIL_LENGTH} characters.", field="email"
        )
    try:
        value = unicodedata.normalize("NFKC", email).strip().casefold()
    except (UnicodeError, TypeError):
        raise AccountValidationError("email_invalid", "Enter a valid email address.", field="email") from None
    if not value:
        raise AccountValidationError("email_required", "Email is required.", field="email")
    if len(value) > MAX_EMAIL_LENGTH:
        raise AccountValidationError(
            "email_too_long", f"Email must be at most {MAX_EMAIL_LENGTH} characters.", field="email"
        )
    if "\x00" in value or EMAIL_PATTERN.fullmatch(value) is None:
        raise AccountValidationError("email_invalid", "Enter a valid email address.", field="email")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise AccountValidationError("email_invalid", "Enter a valid email address.", field="email") from None
    return value


def validate_password(password: str) -> str:
    """Validate a password without modifying or disclosing it."""
    if not isinstance(password, str):
        raise AccountValidationError("password_type", "Password must be a string.", field="password")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AccountValidationError(
            "password_too_long", f"Password must be at most {MAX_PASSWORD_LENGTH} characters.", field="password"
        )
    if "\x00" in password:
        raise AccountValidationError("password_invalid", "Password contains an invalid character.", field="password")
    try:
        password.encode("utf-8")
    except UnicodeEncodeError:
        raise AccountValidationError("password_invalid", "Password contains an invalid character.", field="password") from None
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountValidationError(
            "password_too_short", f"Password must be at least {MIN_PASSWORD_LENGTH} characters.", field="password"
        )
    return password


def account_scope_id(user_id: str, flask_secret: str | bytes) -> str:
    """Derive a deterministic, domain-separated 32-hex account scope."""
    if not isinstance(user_id, str) or ACCOUNT_ID_PATTERN.fullmatch(user_id) is None:
        raise AccountValidationError("user_id_invalid", "User id is invalid.", field="user_id")
    secret = _secret_bytes(flask_secret)
    payload = SCOPE_HMAC_DOMAIN + user_id.encode("ascii")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()[:32]


def _secret_bytes(flask_secret: str | bytes) -> bytes:
    """Validate and normalize an application secret for HMAC use."""
    if isinstance(flask_secret, str):
        try:
            secret = flask_secret.encode("utf-8")
        except UnicodeEncodeError:
            raise AccountValidationError("secret_invalid", "Application secret is invalid.") from None
    elif isinstance(flask_secret, bytes):
        secret = flask_secret
    else:
        raise AccountValidationError("secret_invalid", "Application secret is invalid.")
    if not secret or len(secret) > 4096:
        raise AccountValidationError("secret_invalid", "Application secret is invalid.")
    return secret


def _utc_iso(seconds: float) -> str:
    try:
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="microseconds")
    except (OverflowError, OSError, ValueError):
        raise RuntimeError("Account clock returned an out-of-range value.") from None


def _clock_seconds(clock: Callable[[], Any]) -> float:
    value = clock()
    if isinstance(value, bool):
        raise RuntimeError("Account clock returned an invalid value.")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        try:
            result = value.timestamp()
        except (OverflowError, OSError, ValueError):
            raise RuntimeError("Account clock returned an out-of-range value.") from None
    else:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError("Account clock returned an invalid value.") from None
    if result != result or result in (float("inf"), float("-inf")):
        raise RuntimeError("Account clock returned an invalid value.")
    try:
        datetime.fromtimestamp(result, timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise RuntimeError("Account clock returned an out-of-range value.") from None
    return result


def _safe_int(value: Any, default: int = 0, *, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if result < 0:
        return default
    return min(result, maximum) if maximum is not None else result


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def _sql_norm(sql: Any) -> str:
    return " ".join(str(sql or "").lower().split()).replace("( ", "(").replace(" )", ")")


_ACCOUNTS_SQL = _sql_norm(
    """CREATE TABLE accounts (
        id TEXT NOT NULL PRIMARY KEY CHECK (length(id) = 32 AND id NOT GLOB '*[^0-9a-f]*'),
        email TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login_at TEXT
    )"""
)
_THROTTLE_SQL = _sql_norm(
    """CREATE TABLE login_throttle (
        email_key TEXT NOT NULL PRIMARY KEY CHECK (
            length(email_key) = 64 AND email_key NOT GLOB '*[^0-9a-f]*'
        ),
        failure_count INTEGER NOT NULL,
        window_started_at REAL NOT NULL,
        locked_until REAL NOT NULL DEFAULT 0
    )"""
)


class AccountStore:
    """Persist accounts and bounded login throttling in one SQLite database."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Callable[[], Any] = time.time,
        max_failures: int = DEFAULT_MAX_FAILURES,
        throttle_window_seconds: int = DEFAULT_THROTTLE_WINDOW_SECONDS,
        lockout_seconds: int = DEFAULT_LOCKOUT_SECONDS,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if not isinstance(max_failures, int) or isinstance(max_failures, bool) or not 1 <= max_failures <= 100:
            raise ValueError("max_failures must be an integer from 1 through 100.")
        if not isinstance(throttle_window_seconds, int) or not 1 <= throttle_window_seconds <= 86_400:
            raise ValueError("throttle_window_seconds must be from 1 through 86400.")
        if not isinstance(lockout_seconds, int) or not 1 <= lockout_seconds <= 86_400:
            raise ValueError("lockout_seconds must be from 1 through 86400.")
        if not isinstance(busy_timeout_ms, int) or isinstance(busy_timeout_ms, bool) or not 0 <= busy_timeout_ms <= 120_000:
            raise ValueError("busy_timeout_ms is out of bounds.")
        self.database_path = Path(database_path) if database_path != ":memory:" else None
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_uri = f"file:vibedash-accounts-{id(self)}?mode=memory&cache=shared" if self.database_path is None else None
        self._memory_anchor: sqlite3.Connection | None = None
        self._closed = False
        self.clock = clock
        self.max_failures = max_failures
        self.throttle_window_seconds = throttle_window_seconds
        self.lockout_seconds = lockout_seconds
        self.busy_timeout_ms = busy_timeout_ms
        self._dummy_hash = generate_password_hash(secrets.token_urlsafe(24))
        self._initialize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        anchor, self._memory_anchor = self._memory_anchor, None
        if anchor is not None:
            anchor.close()

    def __enter__(self) -> "AccountStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def register(
        self,
        email: str,
        password: str,
        *,
        credential_secret: str | bytes | None = None,
    ) -> dict[str, Any] | None:
        """Create an account, returning its public record or ``None`` on conflict."""
        normalized_email = normalize_email(email)
        validated_password = validate_password(password)
        secret = _secret_bytes(credential_secret) if credential_secret is not None else None
        account_id = secrets.token_hex(16)
        now = _clock_seconds(self.clock)
        password_hash = generate_password_hash(validated_password)
        try:
            with self._transaction() as connection:
                connection.execute(
                    """INSERT INTO accounts
                       (id, email, password_hash, created_at, last_login_at)
                       VALUES (?, ?, ?, ?, NULL)""",
                    (account_id, normalized_email, password_hash, _utc_iso(now)),
                )
                row = connection.execute(
                    "SELECT id, email, created_at, last_login_at FROM accounts WHERE id = ?", (account_id,)
                ).fetchone()
        except sqlite3.IntegrityError as error:
            if "unique" not in str(error).lower() and "constraint" not in str(error).lower():
                raise
            return None
        account = self._public_account(row)
        return self._attach_credential(account, account_id, password_hash, secret)

    def credential_token(self, account_id: str, flask_secret: str | bytes) -> str | None:
        """Return the opaque session credential for an account.

        The token is an HMAC over the account id and its current password hash.
        Consequently, changing the password invalidates every token minted
        from the previous hash without requiring a schema column or a token
        table.  Only a fixed-format digest leaves this boundary; the hash is
        never returned or logged.
        """
        if not self._valid_stored_id(account_id):
            raise AccountValidationError("user_id_invalid", "User id is invalid.", field="user_id")
        secret = _secret_bytes(flask_secret)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT password_hash FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
        if row is None:
            return None
        password_hash = row["password_hash"]
        return self._credential_digest(account_id, password_hash, secret)

    # The longer name reads better at call sites that keep multiple kinds of
    # session token.  Keep one implementation so the invalidation semantics
    # cannot drift between callers.
    session_credential_token = credential_token

    def verify_credential_token(
        self, account_id: str, token: str, flask_secret: str | bytes
    ) -> bool:
        """Check a session credential token without exposing account data."""
        return self.account_for_credential(account_id, token, flask_secret) is not None

    def account_for_credential(
        self, account_id: str, token: str, flask_secret: str | bytes
    ) -> dict[str, Any] | None:
        """Return the public account represented by a current credential.

        The account row and password hash are read exactly once.  The token is
        derived and compared against that same snapshot before the hash-free
        public record is returned, avoiding a verify-then-get race during
        password rotation.
        """
        if not isinstance(token, str) or CREDENTIAL_TOKEN_PATTERN.fullmatch(token) is None:
            return None
        if not self._valid_stored_id(account_id):
            return None
        secret = _secret_bytes(flask_secret)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, email, password_hash, created_at, last_login_at "
                "FROM accounts WHERE id = ?",
                (account_id,),
            ).fetchone()
        if row is None:
            return None
        password_hash = row["password_hash"]
        expected = self._credential_digest(account_id, password_hash, secret)
        if expected is None:
            return None
        if not hmac.compare_digest(expected, token):
            return None
        return self._public_account(row)

    def change_password(
        self,
        account_id: str,
        current_password: str,
        new_password: str,
        *,
        credential_secret: str | bytes | None = None,
    ) -> dict[str, Any] | None:
        """Change an account password atomically, or return ``None`` safely.

        Missing accounts, invalid account ids, wrong current passwords, and
        malformed stored hashes intentionally share the same generic failure
        result.  An unchanged password is rejected as a no-op.  New-password
        validation remains explicit so callers can give useful form feedback.
        """
        validated_password = validate_password(new_password)
        secret = _secret_bytes(credential_secret) if credential_secret is not None else None
        if not self._valid_stored_id(account_id):
            return None
        now = _clock_seconds(self.clock)
        with self._transaction() as connection:
            account = connection.execute(
                "SELECT id, email, password_hash, created_at, last_login_at "
                "FROM accounts WHERE id = ?",
                (account_id,),
            ).fetchone()
            password_hash = account["password_hash"] if account is not None else self._dummy_hash
            email_key = self._throttle_key(account["email"]) if account is not None else None
            throttle = (
                connection.execute(
                    "SELECT failure_count, window_started_at, locked_until "
                    "FROM login_throttle WHERE email_key = ?",
                    (email_key,),
                ).fetchone()
                if email_key is not None
                else None
            )
            locked = throttle is not None and _safe_float(throttle["locked_until"]) > now
            current_is_well_formed = self._valid_password_input(current_password)
            password_candidate = current_password if current_is_well_formed else ""
            password_ok = self._check_password(password_hash, password_candidate) and current_is_well_formed
            if locked or not password_ok:
                if not locked and email_key is not None:
                    self._record_failure(connection, email_key, now, throttle)
                return None
            # Explicit same-password policy: a successful verification is not
            # enough to rotate a hash and all derived session credentials.
            if isinstance(current_password, str) and hmac.compare_digest(current_password, validated_password):
                return None
            if account is None:
                return None
            new_hash = generate_password_hash(validated_password)
            # BEGIN IMMEDIATE serializes writers; the old-hash predicate also
            # protects this update if the transaction strategy changes later.
            result = connection.execute(
                "UPDATE accounts SET password_hash = ? WHERE id = ? AND password_hash = ?",
                (new_hash, account_id, password_hash),
            )
            if result.rowcount != 1:
                return None
            connection.execute("DELETE FROM login_throttle WHERE email_key = ?", (email_key,))
            refreshed = connection.execute(
                "SELECT id, email, created_at, last_login_at FROM accounts WHERE id = ?",
                (account_id,),
            ).fetchone()
            account = self._public_account(refreshed)
            return self._attach_credential(account, account_id, new_hash, secret)

    def authenticate(
        self,
        email: str,
        password: str,
        *,
        credential_secret: str | bytes | None = None,
    ) -> dict[str, Any] | None:
        """Authenticate without revealing whether an email exists."""
        secret = _secret_bytes(credential_secret) if credential_secret is not None else None
        try:
            normalized_email = normalize_email(email)
        except AccountValidationError:
            if isinstance(password, str) and len(password) <= MAX_PASSWORD_LENGTH and "\x00" not in password:
                self._check_password(self._dummy_hash, password)
            return None
        if not isinstance(password, str) or len(password) > MAX_PASSWORD_LENGTH or "\x00" in password:
            return None
        try:
            password.encode("utf-8")
        except UnicodeEncodeError:
            self._check_password(self._dummy_hash, password)
            return None
        now = _clock_seconds(self.clock)
        email_key = self._throttle_key(normalized_email)
        with self._transaction() as connection:
            account = connection.execute(
                "SELECT id, email, password_hash, created_at, last_login_at FROM accounts WHERE email = ?",
                (normalized_email,),
            ).fetchone()
            throttle = connection.execute(
                "SELECT failure_count, window_started_at, locked_until FROM login_throttle WHERE email_key = ?",
                (email_key,),
            ).fetchone()
            locked = throttle is not None and _safe_float(throttle["locked_until"]) > now
            password_hash = account["password_hash"] if account is not None else self._dummy_hash
            password_ok = self._check_password(password_hash, password)
            if not locked and password_ok and account is not None:
                timestamp = _utc_iso(now)
                connection.execute("DELETE FROM login_throttle WHERE email_key = ?", (email_key,))
                connection.execute("UPDATE accounts SET last_login_at = ? WHERE id = ?", (timestamp, account["id"]))
                refreshed = connection.execute(
                    "SELECT id, email, created_at, last_login_at FROM accounts WHERE id = ?", (account["id"],)
                ).fetchone()
                account = self._public_account(refreshed)
                # The credential is derived from the hash that was checked and
                # while BEGIN IMMEDIATE still excludes a concurrent password
                # rotation.  This snapshot must travel with the authentication
                # result; minting it in a later transaction creates a TOCTOU
                # window between password verification and session creation.
                if account is None:
                    return None
                return self._attach_credential(account, account["id"], password_hash, secret)
            self._record_failure(connection, email_key, now, throttle)
            return None

    def get_account(self, user_id: str) -> dict[str, Any] | None:
        """Fetch one public account record."""
        if not self._valid_stored_id(user_id):
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, email, created_at, last_login_at FROM accounts WHERE id = ?", (user_id,)
            ).fetchone()
        return self._public_account(row)

    def throttle_state(self, email: str) -> dict[str, Any] | None:
        """Return bounded, non-sensitive throttle diagnostics."""
        normalized_email = normalize_email(email)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT failure_count, window_started_at, locked_until FROM login_throttle WHERE email_key = ?",
                (self._throttle_key(normalized_email),),
            ).fetchone()
        if row is None:
            return None
        return {
            "failure_count": _safe_int(row["failure_count"], maximum=self.max_failures),
            "window_started_at": _safe_float(row["window_started_at"]),
            "locked_until": _safe_float(row["locked_until"]),
        }

    def _record_failure(self, connection: sqlite3.Connection, email_key: str, now: float, row: sqlite3.Row | None) -> None:
        count = _safe_int(row["failure_count"], maximum=self.max_failures) if row is not None else 0
        window_started = _safe_float(row["window_started_at"], now) if row is not None else now
        locked_until = _safe_float(row["locked_until"]) if row is not None else 0.0
        if now - window_started >= self.throttle_window_seconds or now < window_started:
            count, window_started, locked_until = 0, now, 0.0
        if locked_until > now:
            return
        count = min(count, self.max_failures - 1) + 1
        if count >= self.max_failures:
            locked_until = now + self.lockout_seconds
        connection.execute(
            """INSERT INTO login_throttle (email_key, failure_count, window_started_at, locked_until)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(email_key) DO UPDATE SET
                 failure_count = excluded.failure_count,
                 window_started_at = excluded.window_started_at,
                 locked_until = excluded.locked_until""",
            (email_key, count, window_started, locked_until),
        )
        self._prune_throttle(connection, now, protected_key=email_key)

    def _prune_throttle(self, connection: sqlite3.Connection, now: float, *, protected_key: str | None = None) -> None:
        connection.execute(
            "DELETE FROM login_throttle WHERE locked_until <= ? AND window_started_at < ?",
            (now, now - self.throttle_window_seconds),
        )
        excess = connection.execute("SELECT COUNT(*) FROM login_throttle").fetchone()[0] - MAX_THROTTLE_ROWS
        if excess <= 0:
            return
        where = "WHERE email_key <> ?" if protected_key is not None else ""
        params: tuple[Any, ...] = (protected_key, excess) if protected_key is not None else (excess,)
        connection.execute(
            f"""DELETE FROM login_throttle WHERE email_key IN (
                SELECT email_key FROM login_throttle {where}
                ORDER BY locked_until ASC, window_started_at ASC, email_key ASC LIMIT ?
            )""",
            params,
        )

    @staticmethod
    def _public_account(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        account_id, email = row["id"], row["email"]
        if not isinstance(account_id, str) or ACCOUNT_ID_PATTERN.fullmatch(account_id) is None:
            return None
        if not isinstance(email, str) or not email:
            return None
        return {
            "id": account_id,
            "email": email,
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }

    @staticmethod
    def _attach_credential(
        account: dict[str, Any] | None,
        account_id: str,
        password_hash: Any,
        secret: bytes | None,
    ) -> dict[str, Any] | None:
        """Attach an internal snapshot credential only to auth-operation results."""
        if account is None or secret is None:
            return account
        token = AccountStore._credential_digest(account_id, password_hash, secret)
        if token is None:
            return None
        account[_SESSION_CREDENTIAL_KEY] = token
        return account

    @staticmethod
    def _valid_password_input(password: Any) -> bool:
        if not isinstance(password, str) or len(password) > MAX_PASSWORD_LENGTH or "\x00" in password:
            return False
        try:
            password.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True

    @staticmethod
    def _valid_password_hash_shape(password_hash: Any) -> bool:
        if not isinstance(password_hash, str) or not 1 <= len(password_hash) <= 4096:
            return False
        parts = password_hash.split("$")
        if len(parts) != 3 or not all(parts):
            return False

        method, salt, digest = parts
        method_parts = method.split(":")
        algorithm = method_parts[0]
        if not algorithm or not salt or not re.fullmatch(r"[0-9a-fA-F]+", digest):
            return False

        if algorithm == "scrypt":
            # Werkzeug emits either the default method name or explicit
            # n/r/p parameters.  Reject malformed parameters before a token
            # can be minted for a hash that password verification cannot use.
            if len(method_parts) == 1:
                parameters = (2**15, 8, 1)
            elif len(method_parts) == 4:
                try:
                    parameters = tuple(int(value) for value in method_parts[1:])
                except (TypeError, ValueError, OverflowError):
                    return False
            else:
                return False
            n, r, p = parameters
            if (
                n < 2
                or n & (n - 1)
                or r < 1
                or p < 1
                or n * r * p > MAX_SCRYPT_MEMORY_BYTES // 132
            ):
                return False
            return len(digest) == 128

        if algorithm == "pbkdf2":
            # Werkzeug supports pbkdf2[:hash_name[:iterations]].
            if len(method_parts) == 1:
                hash_name, iterations = "sha256", 1_000_000
            elif len(method_parts) == 2:
                hash_name, iterations = method_parts[1], 1_000_000
            elif len(method_parts) == 3:
                hash_name = method_parts[1]
                try:
                    iterations = int(method_parts[2])
                except (TypeError, ValueError, OverflowError):
                    return False
            else:
                return False
            if iterations < 1 or iterations > MAX_PBKDF2_ITERATIONS:
                return False
            try:
                # hashlib.new() accepts digest names that pbkdf2_hmac does
                # not.  Probe the exact primitive Werkzeug calls, without
                # doing attacker-controlled work (one iteration only).
                digest_size = len(hashlib.pbkdf2_hmac(hash_name, b"", b"", 1))
            except (TypeError, ValueError, OverflowError):
                return False
            return len(digest) == digest_size * 2

        # md5/sha1/plain were accepted by older Werkzeug releases but are not
        # valid methods for the currently supported checker.  In particular,
        # never mint a credential from an arbitrary three-part legacy-looking
        # string when password verification would fail closed.
        return False

    @staticmethod
    def _credential_digest(account_id: str, password_hash: Any, secret: bytes) -> str | None:
        if not AccountStore._valid_password_hash_shape(password_hash):
            return None
        try:
            hash_bytes = password_hash.encode("utf-8")
        except (AttributeError, UnicodeError):
            return None
        payload = CREDENTIAL_HMAC_DOMAIN + account_id.encode("ascii") + b"\x00" + hash_bytes
        return hmac.new(secret, payload, hashlib.sha256).hexdigest()

    def _check_password(self, password_hash: Any, password: str) -> bool:
        if self._valid_password_hash_shape(password_hash):
            try:
                return bool(check_password_hash(password_hash, password))
            except (ValueError, TypeError, UnicodeError, OverflowError, MemoryError, RuntimeError):
                pass
        try:
            check_password_hash(self._dummy_hash, password)
        except (ValueError, TypeError, UnicodeError, OverflowError, MemoryError, RuntimeError):
            pass
        return False

    @staticmethod
    def _valid_stored_id(value: Any) -> bool:
        return isinstance(value, str) and ACCOUNT_ID_PATTERN.fullmatch(value) is not None

    @staticmethod
    def _throttle_key(normalized_email: str) -> str:
        return hashlib.sha256(normalized_email.encode("utf-8")).hexdigest()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise RuntimeError("Account store is closed.")
        target = str(self.database_path) if self.database_path is not None else self._memory_uri
        connection = sqlite3.connect(
            target, timeout=self.busy_timeout_ms / 1000, uri=self.database_path is None, isolation_level=None
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                try:
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

    def _initialize(self) -> None:
        try:
            if self.database_path is None:
                self._memory_anchor = sqlite3.connect(self._memory_uri, uri=True, isolation_level=None)
                self._memory_anchor.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
                self._memory_anchor.execute("PRAGMA foreign_keys = ON")
            # Hold the write lock while deciding whether the DB is empty. This
            # also makes first-open safe when two workers initialize together.
            with self._transaction() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                objects = connection.execute(
                    "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                ).fetchall()
                if version == 0 and not objects:
                    connection.execute(
                        """CREATE TABLE accounts (
                            id TEXT NOT NULL PRIMARY KEY CHECK (length(id) = 32 AND id NOT GLOB '*[^0-9a-f]*'),
                            email TEXT NOT NULL,
                            password_hash TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            last_login_at TEXT
                        )"""
                    )
                    connection.execute("CREATE UNIQUE INDEX idx_accounts_email_unique ON accounts(email)")
                    connection.execute(
                        """CREATE TABLE login_throttle (
                            email_key TEXT NOT NULL PRIMARY KEY CHECK (
                                length(email_key) = 64 AND email_key NOT GLOB '*[^0-9a-f]*'
                            ),
                            failure_count INTEGER NOT NULL,
                            window_started_at REAL NOT NULL,
                            locked_until REAL NOT NULL DEFAULT 0
                        )"""
                    )
                    connection.execute("CREATE INDEX idx_login_throttle_window ON login_throttle(window_started_at)")
                    connection.execute("PRAGMA user_version = 1")
                elif version != SCHEMA_VERSION:
                    raise AccountStoreSchemaError(
                        f"Account store schema version {version} is unsupported; expected {SCHEMA_VERSION}."
                    )
            self._validate_schema()
        except AccountStoreSchemaError:
            self.close()
            raise
        except (sqlite3.DatabaseError, OSError) as error:
            self.close()
            raise AccountStoreSchemaError(f"Account store schema could not be opened: {error}") from error

    def _validate_schema(self) -> None:
        with self._connection() as connection:
            self._validate_schema_connection(connection)

    @staticmethod
    def _validate_schema_connection(connection: sqlite3.Connection) -> None:
        if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise AccountStoreSchemaError("Account store schema is not v1.")
        objects = connection.execute(
            """SELECT type, name, tbl_name, sql FROM sqlite_master
               WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"""
        ).fetchall()
        expected_objects = {
            ("index", "idx_accounts_email_unique", "accounts"),
            ("index", "idx_login_throttle_window", "login_throttle"),
            ("table", "accounts", "accounts"),
            ("table", "login_throttle", "login_throttle"),
        }
        actual_objects = {(row["type"], row["name"], row["tbl_name"]) for row in objects}
        if actual_objects != expected_objects:
            raise AccountStoreSchemaError("Account store schema objects do not match v1.")
        sql_by_name = {row["name"]: _sql_norm(row["sql"]) for row in objects}
        if sql_by_name["accounts"] != _ACCOUNTS_SQL or sql_by_name["login_throttle"] != _THROTTLE_SQL:
            raise AccountStoreSchemaError("Account store table constraints do not match v1.")
        if sql_by_name["idx_accounts_email_unique"] != _sql_norm(
            "CREATE UNIQUE INDEX idx_accounts_email_unique ON accounts(email)"
        ) or sql_by_name["idx_login_throttle_window"] != _sql_norm(
            "CREATE INDEX idx_login_throttle_window ON login_throttle(window_started_at)"
        ):
            raise AccountStoreSchemaError("Account store indexes do not match v1.")
        AccountStore._expect_columns(
            connection,
            "accounts",
            [("id", "TEXT", 1, None, 1), ("email", "TEXT", 1, None, 0),
             ("password_hash", "TEXT", 1, None, 0), ("created_at", "TEXT", 1, None, 0),
             ("last_login_at", "TEXT", 0, None, 0)],
        )
        AccountStore._expect_columns(
            connection,
            "login_throttle",
            [("email_key", "TEXT", 1, None, 1), ("failure_count", "INTEGER", 1, None, 0),
             ("window_started_at", "REAL", 1, None, 0), ("locked_until", "REAL", 1, "0", 0)],
        )
        AccountStore._expect_index(connection, "accounts", "idx_accounts_email_unique", unique=True, column="email")
        AccountStore._expect_index(connection, "login_throttle", "idx_login_throttle_window", unique=False, column="window_started_at")

    @staticmethod
    def _expect_columns(
        connection: sqlite3.Connection,
        table: str,
        expected: list[tuple[str, str, int, str | None, int]],
    ) -> None:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        actual = [(row[1], row[2].upper(), row[3], row[4], row[5]) for row in rows]
        if actual != expected:
            raise AccountStoreSchemaError(f"Account store table {table!r} columns do not match v1.")

    @staticmethod
    def _expect_index(connection: sqlite3.Connection, table: str, name: str, *, unique: bool, column: str) -> None:
        rows = connection.execute(f"PRAGMA index_list({table})").fetchall()
        matches = [row for row in rows if row[1] == name]
        if len(matches) != 1 or bool(matches[0][2]) != unique:
            raise AccountStoreSchemaError(f"Account store index {name!r} does not match v1.")
        columns = [row[2] for row in connection.execute(f'PRAGMA index_info("{name}")').fetchall()]
        if columns != [column]:
            raise AccountStoreSchemaError(f"Account store index {name!r} does not match v1.")


def validate_account_database(database_path: str | Path) -> None:
    """Validate an existing account database without opening it for writing."""
    path = Path(database_path)
    if path.is_symlink() or not path.is_file():
        raise AccountStoreSchemaError("Account store database is missing or is a symbolic link.")
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            AccountStore._validate_schema_connection(connection)
    except AccountStoreSchemaError:
        raise
    except (sqlite3.DatabaseError, OSError, ValueError) as error:
        raise AccountStoreSchemaError("Account store schema could not be read.") from error


__all__ = [
    "ACCOUNT_ID_PATTERN",
    "AccountStore",
    "AccountStoreSchemaError",
    "AccountValidationError",
    "CREDENTIAL_TOKEN_PATTERN",
    "MAX_EMAIL_LENGTH",
    "MAX_PASSWORD_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "account_scope_id",
    "normalize_email",
    "validate_account_database",
    "validate_password",
]
