import sqlite3
import threading
import unittest
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app
from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.accounts import (
    ACCOUNT_ID_PATTERN,
    MAX_THROTTLE_ROWS,
    AccountStore,
    AccountStoreSchemaError,
    AccountValidationError,
    account_scope_id,
    normalize_email,
)


class FakeClock:
    def __init__(self, value=1_000.0):
        self.value = value

    def __call__(self):
        return self.value


class AccountStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / "accounts.sqlite3"
        self.clock = FakeClock()
        self.store = AccountStore(
            self.path, clock=self.clock, max_failures=3, throttle_window_seconds=10, lockout_seconds=20
        )

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_v1_schema_is_transactional_exact_and_reopen_is_idempotent(self):
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            objects = connection.execute(
                "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
        self.assertEqual(
            objects,
            [("index", "idx_accounts_email_unique"), ("index", "idx_login_throttle_window"),
             ("table", "accounts"), ("table", "login_throttle")],
        )
        self.store.close()
        reopened = AccountStore(self.path, clock=self.clock)
        self.assertIsNotNone(reopened.register("reopen@example.com", "a sufficiently long password"))
        reopened.close()
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_partial_legacy_version_and_constraint_shapes_fail_closed(self):
        self.store.close()
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE accounts")
        with self.assertRaises(AccountStoreSchemaError):
            AccountStore(self.path)

        version_path = Path(self.directory.name) / "version.sqlite3"
        AccountStore(version_path).close()
        with sqlite3.connect(version_path) as connection:
            connection.execute("PRAGMA user_version = 2")
        with self.assertRaisesRegex(AccountStoreSchemaError, "version"):
            AccountStore(version_path)

        extra_path = Path(self.directory.name) / "extra.sqlite3"
        AccountStore(extra_path).close()
        with sqlite3.connect(extra_path) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
        with self.assertRaises(AccountStoreSchemaError):
            AccountStore(extra_path)

        index_path = Path(self.directory.name) / "index.sqlite3"
        AccountStore(index_path).close()
        with sqlite3.connect(index_path) as connection:
            connection.execute("DROP INDEX idx_accounts_email_unique")
        with self.assertRaises(AccountStoreSchemaError):
            AccountStore(index_path)

    def test_normalization_validation_and_public_record(self):
        account = self.store.register("  User@Example.COM ", "a sufficiently long password")
        self.assertEqual(account["email"], "user@example.com")
        self.assertRegex(account["id"], ACCOUNT_ID_PATTERN)
        self.assertNotIn("password_hash", account)
        self.assertIsNone(self.store.register("USER@example.com", "a sufficiently long password"))
        with self.assertRaisesRegex(AccountValidationError, "at least") as raised:
            self.store.register("short@example.com", "short")
        self.assertEqual(raised.exception.code, "password_too_short")
        with self.assertRaises(AccountValidationError):
            self.store.register("not-an-email", "a sufficiently long password")
        with self.assertRaises(AccountValidationError):
            self.store.register("u" * 255 + "@example.com", "a sufficiently long password")
        with self.assertRaises(AccountValidationError):
            self.store.register("user@example.com", "p" * 1025)
        with self.assertRaises(AccountValidationError):
            normalize_email("\ud800@example.com")

    def test_concurrent_duplicate_registration_has_one_winner(self):
        stores = [AccountStore(self.path, clock=self.clock), AccountStore(self.path, clock=self.clock)]
        barrier = threading.Barrier(2)
        results = []

        def register(store):
            barrier.wait()
            results.append(store.register("race@example.com", "a sufficiently long password"))

        threads = [threading.Thread(target=register, args=(store,)) for store in stores]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for store in stores:
            store.close()
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_authentication_is_generic_and_updates_last_login(self):
        account = self.store.register("user@example.com", "a sufficiently long password")
        self.assertIsNotNone(self.store.authenticate("USER@example.com", "a sufficiently long password"))
        self.assertIsNone(self.store.authenticate("user@example.com", "incorrect password"))
        self.assertIsNone(self.store.authenticate("unknown@example.com", "incorrect password"))
        self.assertIsNotNone(self.store.get_account(account["id"])["last_login_at"])
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM login_throttle WHERE email_key = ?", (self.store._throttle_key("unknown@example.com"),)).fetchone()[0],
                1,
            )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM login_throttle WHERE email_key LIKE '%@%' ").fetchone()[0], 0)

    def test_authenticated_credential_snapshot_cannot_be_replaced_by_concurrent_password_change(self):
        """A login result must never mint a token from a later password hash."""
        password = "a sufficiently long password"
        account = self.store.register("snapshot-race@example.com", password)
        authenticated = self.store.authenticate(
            "snapshot-race@example.com",
            password,
            credential_secret="flask-secret",
        )
        self.assertIsNotNone(authenticated)
        snapshot_token = authenticated["_vibedash_session_credential"]

        changed = self.store.change_password(
            account["id"],
            password,
            "a different long password",
        )
        self.assertIsNotNone(changed)
        # The token bound during authentication is now stale and cannot be
        # mistaken for a token minted after the password rotation.
        self.assertFalse(
            self.store.verify_credential_token(
                account["id"], snapshot_token, "flask-secret"
            )
        )
        self.assertNotEqual(
            snapshot_token,
            self.store.credential_token(account["id"], "flask-secret"),
        )

    def test_credential_token_is_opaque_deterministic_separated_and_validated(self):
        account = self.store.register("token@example.com", "a sufficiently long password")
        token = self.store.credential_token(account["id"], "flask-secret")
        self.assertRegex(token, r"^[0-9a-f]{64}$")
        self.assertNotIn("$", token)
        self.assertEqual(token, self.store.session_credential_token(account["id"], "flask-secret"))
        self.assertTrue(self.store.verify_credential_token(account["id"], token, "flask-secret"))
        self.assertNotEqual(token, self.store.credential_token(account["id"], "other-secret"))
        self.assertFalse(self.store.verify_credential_token(account["id"], token, "other-secret"))
        self.assertFalse(self.store.verify_credential_token("not-an-id", token, "flask-secret"))
        self.assertFalse(self.store.verify_credential_token(account["id"], "not-a-token", "flask-secret"))
        self.assertIsNone(self.store.credential_token("b" * 32, "flask-secret"))
        with self.assertRaises(AccountValidationError):
            self.store.credential_token("not-an-id", "flask-secret")
        with self.assertRaises(AccountValidationError):
            self.store.credential_token(account["id"], "\ud800")

    def test_credential_token_fails_closed_for_malformed_hash(self):
        account = self.store.register("broken-token@example.com", "a sufficiently long password")
        for malformed_hash in ("not-a-werkzeug-hash", "scrypt$foo$bar", "scrypt:1:1:1$foo$" + "0" * 128):
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    "UPDATE accounts SET password_hash = ? WHERE id = ?",
                    (malformed_hash, account["id"]),
                )
            self.assertIsNone(self.store.credential_token(account["id"], "flask-secret"))
            self.assertFalse(self.store.verify_credential_token(account["id"], "0" * 64, "flask-secret"))
            self.assertIsNone(self.store.account_for_credential(account["id"], "0" * 64, "flask-secret"))

    def test_credential_and_password_checks_bound_corrupt_kdf_parameters(self):
        account = self.store.register("kdf-bounds@example.com", "a sufficiently long password")
        malformed_hashes = (
            # Product exceeds the bounded Werkzeug scrypt memory contract.
            "scrypt:1048576:8:1$abcdefghijklmnop$" + "0" * 128,
            # Iteration count exceeds the bounded PBKDF2 CPU contract.
            "pbkdf2:sha256:10000001$abcdefghijklmnop$" + "0" * 64,
            # hashlib.new accepts names that Werkzeug's pbkdf2_hmac does not.
            "pbkdf2:shake_128:1$abcdefghijklmnop$" + "0" * 2,
        )
        for malformed_hash in malformed_hashes:
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    "UPDATE accounts SET password_hash = ? WHERE id = ?",
                    (malformed_hash, account["id"]),
                )
            self.assertFalse(AccountStore._valid_password_hash_shape(malformed_hash))
            self.assertIsNone(self.store.credential_token(account["id"], "flask-secret"))
            with patch("vibedash.accounts.check_password_hash", return_value=True) as check:
                self.assertIsNone(self.store.authenticate("kdf-bounds@example.com", "a sufficiently long password"))
            # Invalid stored hashes always take the fixed dummy-hash path and
            # never reach the attacker-controlled KDF parameters.
            check.assert_called_once_with(self.store._dummy_hash, "a sufficiently long password")

    def test_change_password_rotates_hash_and_invalidates_old_credential(self):
        old_password = "a sufficiently long password"
        new_password = "a different long password"
        account = self.store.register("change@example.com", old_password)
        old_token = self.store.credential_token(account["id"], "flask-secret")
        self.assertEqual(self.store.account_for_credential(account["id"], old_token, "flask-secret"), account)

        self.assertIsNone(self.store.change_password(account["id"], "wrong password", new_password))
        self.assertTrue(self.store.verify_credential_token(account["id"], old_token, "flask-secret"))
        self.assertIsNone(self.store.change_password("f" * 32, old_password, new_password))
        changed = self.store.change_password(account["id"], old_password, new_password)
        self.assertEqual(changed["id"], account["id"])
        self.assertNotIn("password_hash", changed)
        self.assertIsNone(self.store.authenticate("change@example.com", old_password))
        self.assertIsNotNone(self.store.authenticate("change@example.com", new_password))
        new_token = self.store.credential_token(account["id"], "flask-secret")
        self.assertNotEqual(old_token, new_token)
        self.assertFalse(self.store.verify_credential_token(account["id"], old_token, "flask-secret"))
        self.assertTrue(self.store.verify_credential_token(account["id"], new_token, "flask-secret"))
        self.assertIsNone(self.store.account_for_credential(account["id"], old_token, "flask-secret"))
        fresh = self.store.account_for_credential(account["id"], new_token, "flask-secret")
        self.assertEqual(fresh["id"], account["id"])
        self.assertEqual(fresh["email"], account["email"])
        self.assertNotIn("password_hash", fresh)
        self.assertIsNone(self.store.account_for_credential("f" * 32, new_token, "flask-secret"))
        self.assertIsNone(self.store.account_for_credential(account["id"], "0" * 64, "flask-secret"))

    def test_change_password_rejects_same_password_without_rotation(self):
        password = "a sufficiently long password"
        account = self.store.register("same@example.com", password)
        token = self.store.credential_token(account["id"], "flask-secret")
        self.assertIsNone(self.store.change_password(account["id"], password, password))
        self.assertEqual(token, self.store.credential_token(account["id"], "flask-secret"))
        self.assertIsNotNone(self.store.authenticate("same@example.com", password))

    def test_change_password_reuses_login_throttle_and_expires_lockout(self):
        password = "a sufficiently long password"
        replacement = "a different long password"
        account = self.store.register("throttled-change@example.com", password)
        for expected_count in (1, 2):
            self.assertIsNone(self.store.change_password(account["id"], "wrong password", replacement))
            self.assertEqual(self.store.throttle_state("throttled-change@example.com")["failure_count"], expected_count)

        self.assertIsNone(self.store.change_password(account["id"], "wrong password", replacement))
        locked_state = self.store.throttle_state("throttled-change@example.com")
        self.assertEqual(locked_state["failure_count"], 3)
        self.assertGreater(locked_state["locked_until"], self.clock.value)
        self.assertIsNone(self.store.change_password(account["id"], password, replacement))
        self.assertIsNotNone(self.store.throttle_state("throttled-change@example.com"))

        self.clock.value += 21
        self.assertIsNotNone(self.store.change_password(account["id"], password, replacement))
        self.assertIsNone(self.store.throttle_state("throttled-change@example.com"))

    def test_change_password_success_clears_non_locked_failures(self):
        password = "a sufficiently long password"
        account = self.store.register("clear-change@example.com", password)
        self.assertIsNone(self.store.change_password(account["id"], "wrong password", "a new long password"))
        self.assertEqual(self.store.throttle_state("clear-change@example.com")["failure_count"], 1)
        self.assertIsNotNone(self.store.change_password(account["id"], password, "a new long password"))
        self.assertIsNone(self.store.throttle_state("clear-change@example.com"))

    def test_same_password_noop_does_not_increment_change_throttle(self):
        password = "a sufficiently long password"
        account = self.store.register("same-throttle@example.com", password)
        self.assertIsNone(self.store.change_password(account["id"], "wrong password", password))
        before = self.store.throttle_state("same-throttle@example.com")
        self.assertEqual(before["failure_count"], 1)
        self.assertIsNone(self.store.change_password(account["id"], password, password))
        self.assertEqual(self.store.throttle_state("same-throttle@example.com"), before)

    def test_change_password_malformed_hash_uses_dummy_and_missing_is_generic(self):
        account = self.store.register("broken-change@example.com", "a sufficiently long password")
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE accounts SET password_hash = ? WHERE id = ?",
                ("not-a-werkzeug-hash", account["id"]),
            )
        with patch("vibedash.accounts.check_password_hash", return_value=True) as check:
            self.assertIsNone(
                self.store.change_password(account["id"], "a sufficiently long password", "a new long password")
            )
        check.assert_called_once_with(self.store._dummy_hash, "a sufficiently long password")
        self.assertIsNone(
            self.store.change_password("e" * 32, "a sufficiently long password", "a new long password")
        )

    def test_change_password_rolls_back_when_hash_generation_fails(self):
        password = "a sufficiently long password"
        account = self.store.register("rollback@example.com", password)
        with patch("vibedash.accounts.generate_password_hash", side_effect=RuntimeError("hash failure")):
            with self.assertRaisesRegex(RuntimeError, "hash failure"):
                self.store.change_password(account["id"], password, "a different long password")
        self.assertIsNotNone(self.store.authenticate("rollback@example.com", password))

    def test_concurrent_password_changes_have_one_winner(self):
        password = "a sufficiently long password"
        stores = [AccountStore(self.path), AccountStore(self.path)]
        account = self.store.register("password-race@example.com", password)
        barrier = threading.Barrier(2)
        results = []
        passwords = ["a first replacement password", "a second replacement password"]

        def change(store, replacement):
            barrier.wait()
            results.append(store.change_password(account["id"], password, replacement))

        threads = [threading.Thread(target=change, args=(store, replacement)) for store, replacement in zip(stores, passwords)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for store in stores:
            store.close()
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(
            sum(self.store.authenticate("password-race@example.com", candidate) is not None for candidate in passwords),
            1,
        )

    def test_malformed_hash_uses_dummy_hash_path(self):
        account = self.store.register("broken@example.com", "a sufficiently long password")
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE accounts SET password_hash = ? WHERE id = ?", ("not-a-werkzeug-hash", account["id"]))
        with patch("vibedash.accounts.check_password_hash", return_value=True) as check:
            self.assertIsNone(self.store.authenticate("broken@example.com", "a sufficiently long password"))
        check.assert_called_once_with(self.store._dummy_hash, "a sufficiently long password")

    def test_lockout_window_reset_and_corrupt_row_hardening(self):
        self.store.register("user@example.com", "a sufficiently long password")
        for _ in range(3):
            self.assertIsNone(self.store.authenticate("user@example.com", "incorrect password"))
        self.assertIsNone(self.store.authenticate("user@example.com", "a sufficiently long password"))
        self.clock.value += 21
        self.assertIsNotNone(self.store.authenticate("user@example.com", "a sufficiently long password"))
        self.assertIsNone(self.store.throttle_state("user@example.com"))

        key = self.store._throttle_key("user@example.com")
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO login_throttle(email_key, failure_count, window_started_at, locked_until) VALUES (?, ?, ?, ?)",
                (key, "9" * 1000, "not-a-time", "not-a-time"),
            )
        self.assertIsNone(self.store.authenticate("user@example.com", "wrong password"))
        state = self.store.throttle_state("user@example.com")
        self.assertGreaterEqual(state["failure_count"], 1)
        self.assertLessEqual(state["failure_count"], self.store.max_failures)

    def test_throttle_state_caps_corrupt_failure_count(self):
        key = self.store._throttle_key("corrupt@example.com")
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO login_throttle(email_key, failure_count, window_started_at, locked_until) VALUES (?, ?, ?, ?)",
                (key, 2**63 - 1, self.clock.value, 0),
            )
        state = self.store.throttle_state("corrupt@example.com")
        self.assertEqual(state["failure_count"], self.store.max_failures)

    def test_throttle_cap_keeps_newly_locked_email(self):
        self.store.register("victim@example.com", "a sufficiently long password")
        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                "INSERT INTO login_throttle(email_key, failure_count, window_started_at, locked_until) VALUES (?, ?, ?, ?)",
                ((f"{index:064x}", 1, self.clock.value, 10**12) for index in range(MAX_THROTTLE_ROWS)),
            )
        for _ in range(self.store.max_failures):
            self.assertIsNone(self.store.authenticate("victim@example.com", "wrong password"))
        state = self.store.throttle_state("victim@example.com")
        self.assertEqual(state["failure_count"], self.store.max_failures)
        self.assertGreater(state["locked_until"], self.clock.value)

    def test_memory_lifecycle_and_busy_timeout(self):
        store = AccountStore(":memory:", clock=self.clock)
        store.register("memory@example.com", "a sufficiently long password")
        with store._connection() as connection:
            self.assertEqual(connection.execute("PRAGMA busy_timeout").fetchone()[0], 5000)
        anchor = store._memory_anchor
        store.close()
        self.assertIsNone(store._memory_anchor)
        with self.assertRaises(sqlite3.ProgrammingError):
            anchor.execute("SELECT 1")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            store.get_account("a" * 32)
        with AccountStore(":memory:") as managed:
            self.assertIsNotNone(managed.register("context@example.com", "a sufficiently long password"))

    def test_clock_failure_rolls_back_registration(self):
        store = AccountStore(Path(self.directory.name) / "clock.sqlite3", clock=lambda: 10**100)
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            store.register("clock@example.com", "a sufficiently long password")
        self.assertIsNone(store.get_account("a" * 32))


class AccountScopeTests(unittest.TestCase):
    def test_scope_is_deterministic_separated_and_validated(self):
        user_id = "a" * 32
        first = account_scope_id(user_id, "secret")
        self.assertEqual(first, account_scope_id(user_id, "secret"))
        self.assertNotEqual(first, account_scope_id(user_id, "other-secret"))
        self.assertRegex(first, r"^[0-9a-f]{32}$")
        with self.assertRaises(AccountValidationError):
            account_scope_id("not-an-id", "secret")
        with self.assertRaises(AccountValidationError):
            account_scope_id(user_id, "\ud800")


class AccountRouteScopeTests(unittest.TestCase):
    def test_malformed_application_secret_discards_account_identity(self):
        import web_app
        from vibedash.routes import _analysis_scope_id

        with TemporaryDirectory() as directory:
            previous_path = web_app.app.config["VIBEDASH_ACCOUNT_STORE_PATH"]
            previous_secret = web_app.app.secret_key
            web_app.app.config["VIBEDASH_ACCOUNT_STORE_PATH"] = str(Path(directory) / "accounts.sqlite3")
            web_app.app.extensions.pop("vibedash_account_store", None)
            try:
                with web_app.app.test_request_context("/vibedash/"):
                    store = AccountStore(web_app.app.config["VIBEDASH_ACCOUNT_STORE_PATH"])
                    account = store.register("scope@example.com", "a sufficiently long password")
                    web_app.app.extensions["vibedash_account_store"] = store
                    from flask import session

                    session["vibedash_account_id"] = account["id"]
                    session["vibedash_analysis_scope_id"] = "b" * 32
                    session["vibedash_auth_csrf_token"] = "c" * 64
                    session["vibedash_decision_csrf_token"] = "d" * 64
                    web_app.app.secret_key = None
                    scope_id = _analysis_scope_id()
                    self.assertRegex(scope_id, re.compile(r"^[0-9a-f]{32}$"))
                    self.assertNotEqual(scope_id, "b" * 32)
                    self.assertNotIn("vibedash_account_id", session)
                    self.assertNotIn("vibedash_auth_csrf_token", session)
                    self.assertNotIn("vibedash_decision_csrf_token", session)
                    self.assertEqual(session["vibedash_analysis_scope_id"], scope_id)
            finally:
                store = web_app.app.extensions.pop("vibedash_account_store", None)
                if store is not None:
                    store.close()
                web_app.app.config["VIBEDASH_ACCOUNT_STORE_PATH"] = previous_path
                web_app.app.secret_key = previous_secret


class AccountRouteIntegrationTests(unittest.TestCase):
    """Exercise the browser boundary around the durable account store."""

    PASSWORD = "a sufficiently long password"

    def setUp(self):
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        self.account_path = root / "accounts" / "accounts.sqlite3"
        self.job_path = root / "jobs" / "analysis_jobs.sqlite3"
        self.previous_config = {
            key: web_app.app.config[key]
            for key in (
                "TESTING",
                "VIBEDASH_ACCOUNT_STORE_PATH",
                "VIBEDASH_JOB_STORE_PATH",
                "UPLOAD_FOLDER",
                "REPORT_FOLDER",
                "BASELINE_FOLDER",
                "SESSION_COOKIE_SECURE",
            )
        }
        web_app.app.config.update(
            TESTING=True,
            VIBEDASH_ACCOUNT_STORE_PATH=str(self.account_path),
            VIBEDASH_JOB_STORE_PATH=str(self.job_path),
            UPLOAD_FOLDER=str(root / "uploads"),
            REPORT_FOLDER=str(root / "reports"),
            BASELINE_FOLDER=str(root / "baselines"),
            SESSION_COOKIE_SECURE=False,
        )
        for directory in (root / "uploads", root / "reports", root / "baselines"):
            directory.mkdir(parents=True, exist_ok=True)
        old_store = web_app.app.extensions.pop("vibedash_account_store", None)
        if old_store is not None:
            old_store.close()
        self.addCleanup(self._restore_app)
        self.submit_patch = patch(
            "vibedash.routes.analysis_job_dispatcher.submit", return_value=True
        )
        self.submit_patch.start()
        self.addCleanup(self.submit_patch.stop)

    def _restore_app(self):
        store = web_app.app.extensions.pop("vibedash_account_store", None)
        if store is not None:
            store.close()
        web_app.app.config.update(self.previous_config)
        self.directory.cleanup()

    @staticmethod
    def _csrf(client, path):
        response = client.get(path)
        token = re.search(
            r'name="csrf_token" value="([0-9a-f]{64})"',
            response.get_data(as_text=True),
        )
        if token is None:
            raise AssertionError("CSRF token missing from form")
        return token.group(1)

    @staticmethod
    def _logout_csrf(client):
        html = client.get("/vibedash/").get_data(as_text=True)
        token = re.search(
            r'action="/vibedash/logout".*?name="csrf_token" value="([0-9a-f]{64})"',
            html,
            re.DOTALL,
        )
        if token is None:
            raise AssertionError("Logout CSRF token missing from landing page")
        return token.group(1)

    def _register(self, client, email="user@example.com"):
        return client.post(
            "/vibedash/register",
            data={
                "csrf_token": self._csrf(client, "/vibedash/register"),
                "email": email,
                "password": self.PASSWORD,
            },
        )

    def _login(self, client, email="user@example.com", password=None):
        return client.post(
            "/vibedash/login",
            data={
                "csrf_token": self._csrf(client, "/vibedash/login"),
                "email": email,
                "password": password or self.PASSWORD,
            },
        )

    def test_two_clients_share_account_history_logout_denies_and_relogin_restores(self):
        first = web_app.app.test_client()
        second = web_app.app.test_client()
        self.assertEqual(self._register(first).status_code, 302)
        with first.session_transaction() as browser:
            account_id = browser["vibedash_account_id"]
        scope = account_scope_id(account_id, web_app.app.secret_key)
        job = AnalysisJobStore(self.job_path).create(
            scope,
            {"demo_dataset": "saas_growth", "prompt": "shared history"},
        )

        self.assertEqual(self._login(second).status_code, 302)
        self.assertIn(job["id"][:10], first.get("/vibedash/history").get_data(as_text=True))
        self.assertIn(job["id"][:10], second.get("/vibedash/history").get_data(as_text=True))
        stranger = web_app.app.test_client()
        self.assertEqual(stranger.get(f"/vibedash/jobs/{job['id']}").status_code, 404)
        with second.session_transaction() as browser:
            self.assertNotIn("monitoring_scope_id", browser)

        self.assertEqual(first.get("/vibedash/logout").status_code, 405)
        self.assertEqual(first.post("/vibedash/logout").status_code, 400)
        token = self._logout_csrf(first)
        self.assertEqual(
            first.post("/vibedash/logout", data={"csrf_token": token}).status_code,
            302,
        )
        self.assertNotIn(job["id"][:10], first.get("/vibedash/history").get_data(as_text=True))
        self.assertEqual(self._login(first).status_code, 302)
        self.assertIn(job["id"][:10], first.get("/vibedash/history").get_data(as_text=True))

    def test_guest_job_is_not_claimed_and_auth_failures_are_generic(self):
        client = web_app.app.test_client()
        guest_job = client.post(
            "/vibedash/jobs",
            data={"demo_dataset": "saas_growth", "prompt": "guest"},
        ).get_json()["job_id"]
        self.assertEqual(
            client.post(
                "/vibedash/register",
                data={"csrf_token": "bad", "email": "user@example.com", "password": self.PASSWORD},
            ).status_code,
            400,
        )
        self.assertEqual(self._register(client).status_code, 302)
        history = client.get("/vibedash/history").get_data(as_text=True)
        self.assertNotIn(guest_job[:10], history)

        duplicate = web_app.app.test_client()
        duplicate_response = self._register(duplicate)
        duplicate_html = duplicate_response.get_data(as_text=True)
        self.assertEqual(duplicate_response.status_code, 400)
        self.assertIn("Unable to create an account with those details.", duplicate_html)
        self.assertNotIn("already exists", duplicate_html.lower())
        unknown = web_app.app.test_client()
        unknown_response = self._login(unknown, email="unknown@example.com", password="wrong password")
        malformed = web_app.app.test_client()
        malformed_response = self._login(malformed, email="not-an-email", password="wrong password")
        self.assertEqual(unknown_response.status_code, malformed_response.status_code,)
        self.assertIn("Email or password is incorrect.", unknown_response.get_data(as_text=True))
        self.assertIn("Email or password is incorrect.", malformed_response.get_data(as_text=True))

    def test_auth_transitions_preserve_classic_upload_and_monitoring_state(self):
        client = web_app.app.test_client()
        with client.session_transaction() as browser:
            browser['dataset_filename'] = 'classic.csv'
            browser['report_filename'] = 'classic.html'
            browser['monitoring_scope_id'] = 'a' * 32

        self.assertEqual(self._register(client, 'preserve@example.com').status_code, 302)
        with client.session_transaction() as browser:
            self.assertEqual(browser['dataset_filename'], 'classic.csv')
            self.assertEqual(browser['report_filename'], 'classic.html')
            self.assertEqual(browser['monitoring_scope_id'], 'a' * 32)
            self.assertNotIn('vibedash_analysis_scope_id', browser)

        token = self._logout_csrf(client)
        self.assertEqual(client.post('/vibedash/logout', data={'csrf_token': token}).status_code, 302)
        with client.session_transaction() as browser:
            self.assertEqual(browser['dataset_filename'], 'classic.csv')
            self.assertEqual(browser['report_filename'], 'classic.html')
            self.assertEqual(browser['monitoring_scope_id'], 'a' * 32)

        self.assertEqual(self._login(client, 'preserve@example.com').status_code, 302)
        with client.session_transaction() as browser:
            self.assertEqual(browser['dataset_filename'], 'classic.csv')
            self.assertEqual(browser['report_filename'], 'classic.html')
            self.assertEqual(browser['monitoring_scope_id'], 'a' * 32)

    def test_malformed_secret_hides_account_identity_and_cookie_contract_is_explicit(self):
        client = web_app.app.test_client()
        self.assertEqual(self._register(client, "secret@example.com").status_code, 302)
        with client.session_transaction() as browser:
            account_id = browser["vibedash_account_id"]
        with patch("vibedash.routes.account_scope_id", side_effect=AccountValidationError("secret_invalid", "bad secret")):
            landing = client.get("/vibedash/").get_data(as_text=True)
            self.assertNotIn("secret@example.com", landing)
            self.assertIn("Sign in", landing)
            with client.session_transaction() as browser:
                self.assertNotIn("vibedash_account_id", browser)
        self.assertEqual(self._register(client, "missing@example.com").status_code, 302)
        store = web_app.app.extensions.pop("vibedash_account_store")
        store.close()
        original_path = web_app.app.config["VIBEDASH_ACCOUNT_STORE_PATH"]
        try:
            web_app.app.config["VIBEDASH_ACCOUNT_STORE_PATH"] = self.directory.name
            landing = client.get("/vibedash/").get_data(as_text=True)
            self.assertIn("Sign in", landing)
            self.assertNotIn("missing@example.com", landing)
        finally:
            web_app.app.config["VIBEDASH_ACCOUNT_STORE_PATH"] = original_path
        self.assertTrue(web_app.app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(web_app.app.config["SESSION_COOKIE_SAMESITE"], "Lax")
        web_app.app.config["SESSION_COOKIE_SECURE"] = True
        secure_cookie = web_app.app.test_client().get("/vibedash/login").headers.get("Set-Cookie", "")
        self.assertIn("Secure", secure_cookie)

    def test_invalid_account_rotation_mints_fresh_auth_csrf_token(self):
        client = web_app.app.test_client()
        with client.session_transaction() as browser:
            browser["vibedash_account_id"] = "z" * 32
            browser["vibedash_auth_csrf_token"] = "a" * 64
            browser["vibedash_decision_csrf_token"] = "b" * 64
            browser["vibedash_analysis_scope_id"] = "c" * 32

        token = self._csrf(client, "/vibedash/register")
        response = client.post(
            "/vibedash/register",
            data={"csrf_token": token, "email": "fresh@example.com", "password": self.PASSWORD},
        )
        self.assertEqual(response.status_code, 302)

    def test_account_settings_guest_csrf_mismatch_and_explicit_password_validation(self):
        client = web_app.app.test_client()
        self.assertEqual(client.get("/vibedash/account").status_code, 302)
        self.assertEqual(client.post("/vibedash/account/password").status_code, 302)
        self.assertEqual(self._register(client, "settings-validation@example.com").status_code, 302)

        settings = client.get("/vibedash/account")
        token = re.search(r'name="csrf_token" value="([0-9a-f]{64})"', settings.get_data(as_text=True)).group(1)
        mismatch = client.post(
            "/vibedash/account/password",
            data={
                "csrf_token": token,
                "current_password": self.PASSWORD,
                "new_password": "a different long password",
                "new_password_confirmation": "a mismatched long password",
            },
        )
        self.assertEqual(mismatch.status_code, 400)
        self.assertIn("New passwords do not match.", mismatch.get_data(as_text=True))

        # Only the documented confirmation field is accepted; aliases must not
        # accidentally turn into a successful password change.
        alias = client.post(
            "/vibedash/account/password",
            data={
                "csrf_token": token,
                "current_password": self.PASSWORD,
                "new_password": "a different long password",
                "confirm_password": "a different long password",
            },
        )
        self.assertEqual(alias.status_code, 400)
        self.assertIn("New passwords do not match.", alias.get_data(as_text=True))

        short = client.post(
            "/vibedash/account/password",
            data={
                "csrf_token": token,
                "current_password": self.PASSWORD,
                "new_password": "too short",
                "new_password_confirmation": "too short",
            },
        )
        self.assertEqual(short.status_code, 400)
        self.assertIn("at least", short.get_data(as_text=True))
        self.assertEqual(
            client.post(
                "/vibedash/account/password",
                data={
                    "csrf_token": "0" * 64,
                    "current_password": self.PASSWORD,
                    "new_password": "a different long password",
                    "new_password_confirmation": "a different long password",
                },
            ).status_code,
            400,
        )

    def test_account_export_guest_and_csrf_are_rejected_without_export_work(self):
        guest = web_app.app.test_client()
        response = guest.post('/vibedash/account/export.json', data={'csrf_token': 'x'})
        self.assertEqual(response.status_code, 302)
        self.assertIn('/vibedash/login', response.location)

        client = web_app.app.test_client()
        self.assertEqual(self._register(client, 'export-csrf@example.com').status_code, 302)
        before = sorted(path.relative_to(self.directory.name).as_posix() for path in Path(self.directory.name).rglob('*'))
        response = client.post('/vibedash/account/export.json', data={'csrf_token': '0' * 64})
        self.assertEqual(response.status_code, 400)
        self.assertIn('temporarily unavailable', response.get_data(as_text=True))
        after = sorted(path.relative_to(self.directory.name).as_posix() for path in Path(self.directory.name).rglob('*'))
        self.assertEqual(before, after)

    def test_account_export_is_scoped_bounded_and_has_download_headers(self):
        from vibedash import account_export

        client = web_app.app.test_client()
        self.assertEqual(self._register(client, 'export-http@example.com').status_code, 302)
        with client.session_transaction() as browser:
            account_id = browser['vibedash_account_id']

        with patch('vibedash.routes._analysis_job_store') as job_store_factory, patch(
            'vibedash.routes._decision_case_store'
        ) as case_store_factory:
            job_store_factory.return_value.list_for_scope_page.return_value = ([{
                'id': 'job',
                'payload': {'prompt': 'HTTP_PROMPT_SECRET', 'filename': 'HTTP_FILE_SECRET'},
            }], True)
            case_store_factory.return_value.list_for_scope_page.return_value = ([{
                'id': 'case',
                'evidence_snapshot': {'finding': 'HTTP_SNAPSHOT_SECRET'},
            }], False)
            token = self._csrf(client, '/vibedash/account')
            response = client.post('/vibedash/account/export.json', data={'csrf_token': token})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['jobs_truncated'], True)
        self.assertEqual(payload['decisions_truncated'], False)
        for marker in ('HTTP_PROMPT_SECRET', 'HTTP_FILE_SECRET', 'HTTP_SNAPSHOT_SECRET'):
            self.assertNotIn(marker, response.get_data(as_text=True))
        self.assertEqual(response.headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertIn(f'data-prism-account-export-{account_id[:12]}.json', response.headers['Content-Disposition'])
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
        self.assertEqual(response.headers['Pragma'], 'no-cache')
        self.assertEqual(response.headers['Referrer-Policy'], 'no-referrer')
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
        self.assertEqual(response.headers['Content-Security-Policy'], "default-src 'none'; sandbox")
        job_store_factory.return_value.list_for_scope_page.assert_called_once_with(
            account_scope_id(account_id, web_app.app.secret_key),
            limit=account_export.MAX_ACCOUNT_EXPORT_JOBS,
        )
        case_store_factory.return_value.list_for_scope_page.assert_called_once_with(
            account_scope_id(account_id, web_app.app.secret_key),
            limit=account_export.MAX_ACCOUNT_EXPORT_CASES,
        )

    def test_account_export_builder_and_size_failures_are_generic_and_safe(self):
        from vibedash import account_export

        client = web_app.app.test_client()
        self.assertEqual(self._register(client, 'export-failure@example.com').status_code, 302)
        token = self._csrf(client, '/vibedash/account')
        with patch('vibedash.routes.build_account_export', side_effect=RuntimeError('sensitive marker')):
            failed = client.post('/vibedash/account/export.json', data={'csrf_token': token})
        self.assertEqual(failed.status_code, 500)
        self.assertNotIn('sensitive marker', failed.get_data(as_text=True))

        size_error = getattr(
            account_export,
            'AccountExportTooLargeError',
            getattr(account_export, 'AccountExportSizeError', None),
        )
        if size_error is not None:
            token = self._csrf(client, '/vibedash/account')
            with patch('vibedash.routes.build_account_export', side_effect=size_error('too big')):
                oversized = client.post('/vibedash/account/export.json', data={'csrf_token': token})
            self.assertEqual(oversized.status_code, 413)
            self.assertNotIn('too big', oversized.get_data(as_text=True))

    def test_account_export_unknown_failure_is_never_classified_by_name_or_status(self):
        from vibedash import routes

        class LimitLikeRuntimeError(RuntimeError):
            status_code = 413

        self.assertEqual(
            routes._account_export_failure_status(LimitLikeRuntimeError('internal detail')),
            500,
        )

    def test_account_export_isolation_and_stale_credentials_fail_closed(self):
        first = web_app.app.test_client()
        second = web_app.app.test_client()
        self.assertEqual(self._register(first, 'export-first@example.com').status_code, 302)
        self.assertEqual(self._register(second, 'export-second@example.com').status_code, 302)
        with first.session_transaction() as browser:
            first_id = browser['vibedash_account_id']
        with second.session_transaction() as browser:
            second_id = browser['vibedash_account_id']
        job_store = AnalysisJobStore(self.job_path)
        first_job = job_store.create(
            account_scope_id(first_id, web_app.app.secret_key),
            {'demo_dataset': 'first-account'},
        )
        second_job = job_store.create(
            account_scope_id(second_id, web_app.app.secret_key),
            {'demo_dataset': 'second-account'},
        )

        first_token = self._csrf(first, '/vibedash/account')
        first_export = first.post('/vibedash/account/export.json', data={'csrf_token': first_token})
        self.assertEqual(first_export.status_code, 200)
        self.assertIn(first_job['id'], first_export.get_data(as_text=True))
        self.assertNotIn(second_job['id'], first_export.get_data(as_text=True))
        second_token = self._csrf(second, '/vibedash/account')
        second_export = second.post('/vibedash/account/export.json', data={'csrf_token': second_token})
        self.assertEqual(second_export.status_code, 200)
        self.assertIn(second_job['id'], second_export.get_data(as_text=True))
        self.assertNotIn(first_job['id'], second_export.get_data(as_text=True))

        copied_cookie = first.get_cookie('session')
        stale = web_app.app.test_client()
        stale.set_cookie('session', copied_cookie.value)
        account_store = web_app.app.extensions['vibedash_account_store']
        account_store.change_password(
            first_id,
            self.PASSWORD,
            'a different long password',
            credential_secret=web_app.app.secret_key,
        )
        stale_export = stale.post('/vibedash/account/export.json', data={'csrf_token': '0' * 64})
        self.assertEqual(stale_export.status_code, 302)
        self.assertIn('/vibedash/login', stale_export.location)

    def test_account_password_wrong_current_locks_out_and_success_flash_is_visible(self):
        client = web_app.app.test_client()
        self.assertEqual(self._register(client, "settings-lock@example.com").status_code, 302)
        for _ in range(5):
            token = self._csrf(client, "/vibedash/account")
            wrong = client.post(
                "/vibedash/account/password",
                data={
                    "csrf_token": token,
                    "current_password": "wrong current password",
                    "new_password": "a different long password",
                    "new_password_confirmation": "a different long password",
                },
            )
            self.assertEqual(wrong.status_code, 400)
            self.assertIn("current password or account state is invalid", wrong.get_data(as_text=True))
        token = self._csrf(client, "/vibedash/account")
        locked = client.post(
            "/vibedash/account/password",
            data={
                "csrf_token": token,
                "current_password": self.PASSWORD,
                "new_password": "a different long password",
                "new_password_confirmation": "a different long password",
            },
        )
        self.assertEqual(locked.status_code, 400)
        self.assertIn("current password or account state is invalid", locked.get_data(as_text=True))

    def test_password_change_revokes_copied_cookie_and_preserves_current_session(self):
        first = web_app.app.test_client()
        self.assertEqual(self._register(first, "rotate-http@example.com").status_code, 302)
        with first.session_transaction() as browser:
            account_id = browser["vibedash_account_id"]
        shared_scope = account_scope_id(account_id, web_app.app.secret_key)
        job = AnalysisJobStore(self.job_path).create(
            shared_scope,
            {"demo_dataset": "saas_growth", "prompt": "password scope"},
        )
        copied_cookie = first.get_cookie("session")
        second = web_app.app.test_client()
        second.set_cookie("session", copied_cookie.value)
        self.assertEqual(second.get("/vibedash/account").status_code, 200)

        token = self._csrf(first, "/vibedash/account")
        changed = first.post(
            "/vibedash/account/password",
            data={
                "csrf_token": token,
                "current_password": self.PASSWORD,
                "new_password": "a different long password",
                "new_password_confirmation": "a different long password",
            },
            follow_redirects=True,
        )
        self.assertEqual(changed.status_code, 200)
        self.assertIn("Your password was changed.", changed.get_data(as_text=True))
        self.assertEqual(first.get("/vibedash/account").status_code, 200)
        self.assertIn(job["id"][:10], first.get("/vibedash/history").get_data(as_text=True))
        self.assertEqual(second.get("/vibedash/account").status_code, 302)
        self.assertNotIn(job["id"][:10], second.get("/vibedash/history").get_data(as_text=True))

        relogin = web_app.app.test_client()
        old_password = self._login(
            relogin, "rotate-http@example.com", password=self.PASSWORD
        )
        self.assertEqual(old_password.status_code, 401)
        new_password = self._login(
            relogin, "rotate-http@example.com", password="a different long password"
        )
        self.assertEqual(new_password.status_code, 302)

    def test_missing_or_malformed_credential_rotates_vibedash_keys_but_keeps_classic_state(self):
        client = web_app.app.test_client()
        self.assertEqual(self._register(client, "malformed-credential@example.com").status_code, 302)
        with client.session_transaction() as browser:
            browser["dataset_filename"] = "classic.csv"
            browser["report_filename"] = "classic.html"
            browser["monitoring_scope_id"] = "b" * 32
            browser["vibedash_analysis_scope_id"] = "c" * 32
            browser.pop("vibedash_account_credential")
            browser["vibedash_auth_csrf_token"] = "d" * 64
            browser["vibedash_decision_csrf_token"] = "e" * 64
        self.assertEqual(client.get("/vibedash/account").status_code, 302)
        with client.session_transaction() as browser:
            self.assertNotIn("vibedash_account_id", browser)
            self.assertNotIn("vibedash_account_credential", browser)
            self.assertNotEqual(browser.get("vibedash_analysis_scope_id"), "c" * 32)
            self.assertNotIn("vibedash_auth_csrf_token", browser)
            self.assertNotIn("vibedash_decision_csrf_token", browser)
            self.assertEqual(browser["dataset_filename"], "classic.csv")
            self.assertEqual(browser["report_filename"], "classic.html")
            self.assertEqual(browser["monitoring_scope_id"], "b" * 32)

        self.assertEqual(self._login(client, "malformed-credential@example.com").status_code, 302)
        with client.session_transaction() as browser:
            browser["vibedash_account_credential"] = "not-a-token"
            browser["vibedash_analysis_scope_id"] = "f" * 32
        self.assertEqual(client.get("/vibedash/account").status_code, 302)
        with client.session_transaction() as browser:
            self.assertNotIn("vibedash_account_id", browser)
            self.assertNotIn("vibedash_account_credential", browser)
            self.assertNotEqual(browser.get("vibedash_analysis_scope_id"), "f" * 32)
            self.assertEqual(browser["monitoring_scope_id"], "b" * 32)

    def test_token_mint_failure_rotates_identity_without_rendering_stale_account(self):
        client = web_app.app.test_client()
        with client.session_transaction() as browser:
            browser["dataset_filename"] = "classic.csv"
            browser["report_filename"] = "classic.html"
            browser["monitoring_scope_id"] = "a" * 32
        self.assertEqual(self._register(client, "mint-failure@example.com").status_code, 302)
        store = web_app.app.extensions["vibedash_account_store"]
        token = self._csrf(client, "/vibedash/account")
        original_change = store.change_password
        with patch.object(store, "credential_token", return_value=None), patch.object(
            store,
            "change_password",
            side_effect=lambda account_id, current_password, new_password, **kwargs: {
                key: value
                for key, value in (original_change(
                    account_id,
                    current_password,
                    new_password,
                    **kwargs,
                ) or {}).items()
                if key != "_vibedash_session_credential"
            } or None,
        ):
            response = client.post(
                "/vibedash/account/password",
                data={
                    "csrf_token": token,
                    "current_password": self.PASSWORD,
                    "new_password": "a different long password",
                    "new_password_confirmation": "a different long password",
                },
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/vibedash/login")
        with client.session_transaction() as browser:
            self.assertNotIn("vibedash_account_id", browser)
            self.assertNotIn("vibedash_account_credential", browser)
            self.assertNotIn("vibedash_auth_csrf_token", browser)
            self.assertNotIn("vibedash_decision_csrf_token", browser)
            self.assertNotIn("vibedash_analysis_scope_id", browser)
            self.assertEqual(browser["dataset_filename"], "classic.csv")
            self.assertEqual(browser["report_filename"], "classic.html")
            self.assertEqual(browser["monitoring_scope_id"], "a" * 32)

    def test_auth_snapshot_race_cannot_establish_session_for_new_password_hash(self):
        """Interleaving a password rotation must leave the login snapshot stale."""
        store = web_app.app.extensions.get("vibedash_account_store")
        if store is None:
            store = AccountStore(self.account_path)
            web_app.app.extensions["vibedash_account_store"] = store
        account = store.register("snapshot-route-race@example.com", self.PASSWORD)
        authenticated = store.authenticate(
            "snapshot-route-race@example.com",
            self.PASSWORD,
            credential_secret=web_app.app.secret_key,
        )
        self.assertIsNotNone(authenticated)
        self.assertIsNotNone(
            store.change_password(
                account["id"], self.PASSWORD, "a different long password"
            )
        )

        from flask import session
        from vibedash.routes import _current_account, _establish_account

        with web_app.app.test_request_context("/vibedash/login"):
            self.assertTrue(_establish_account(authenticated))
            # The established snapshot is the old token, never a token minted
            # from the replacement hash; the next account resolution therefore
            # rejects and rotates it instead of authorizing the login.
            self.assertIsNone(_current_account())
            self.assertNotIn("vibedash_account_id", session)
            self.assertNotIn("vibedash_account_credential", session)

    def test_account_establishment_session_failure_is_fail_closed(self):
        client = web_app.app.test_client()
        self.assertEqual(self._register(client, "session-failure@example.com").status_code, 302)
        with client.session_transaction() as browser:
            browser["dataset_filename"] = "classic.csv"
            browser["report_filename"] = "classic.html"
            browser["monitoring_scope_id"] = "a" * 32
            account_id = browser["vibedash_account_id"]
            account = {"id": account_id, "email": "session-failure@example.com"}

        from vibedash.routes import _establish_account

        # Simulate a failure after the account id and credential have been
        # written, but before the new session can be considered complete.
        with patch("vibedash.routes._auth_csrf_token", side_effect=RuntimeError("csrf failure")):
            with web_app.app.test_request_context("/vibedash/account"):
                from flask import session

                session.update(
                    {
                        "dataset_filename": "classic.csv",
                        "report_filename": "classic.html",
                        "monitoring_scope_id": "a" * 32,
                    }
                )
                self.assertFalse(_establish_account(account))
                self.assertNotIn("vibedash_account_id", session)
                self.assertNotIn("vibedash_account_credential", session)
                self.assertNotIn("vibedash_auth_csrf_token", session)
                self.assertNotIn("vibedash_decision_csrf_token", session)
                self.assertEqual(session["dataset_filename"], "classic.csv")
                self.assertEqual(session["report_filename"], "classic.html")
                self.assertEqual(session["monitoring_scope_id"], "a" * 32)

    def test_account_establishment_rejects_public_record_without_snapshot_credential(self):
        client = web_app.app.test_client()
        self.assertEqual(self._register(client, "public-record@example.com").status_code, 302)
        with client.session_transaction() as browser:
            account_id = browser["vibedash_account_id"]

        from vibedash.routes import _establish_account

        # A public account record must never trigger a later credential_token
        # read.  Session establishment is valid only for an auth-operation
        # result carrying the exact transaction-bound hash snapshot credential.
        public_account = {"id": account_id, "email": "public-record@example.com"}
        with patch.object(web_app.app.extensions["vibedash_account_store"], "credential_token") as mint:
            with web_app.app.test_request_context("/vibedash/login"):
                from flask import session

                self.assertFalse(_establish_account(public_account))
                mint.assert_not_called()
                self.assertNotIn("vibedash_account_id", session)
                self.assertNotIn("vibedash_account_credential", session)


if __name__ == "__main__":
    unittest.main()
