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


if __name__ == "__main__":
    unittest.main()
