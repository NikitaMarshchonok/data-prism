import unittest
from datetime import datetime, timezone

import requests

from release_check import (
    ReleaseCheckInputError,
    normalize_base_url,
    render_human_report,
    revisions_match,
    verify_release,
)


class FakeResponse:
    def __init__(self, status_code, payload=None, *, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.responses.pop(0)


def successful_responses(request_id="release-check-test"):
    return [
        FakeResponse(
            200,
            {"status": "ok", "service": "data-prism", "version": "e1d2bae12345"},
            headers={"X-Request-ID": request_id, "Content-Type": "application/json"},
        ),
        FakeResponse(
            200,
            {
                "status": "ready",
                "issues": [],
                "warnings": ["Runtime state can be lost during restart."],
                "version": "e1d2bae12345",
                "runtime_storage": {
                    "deployment_profile": "demo",
                    "state_durability": "ephemeral",
                },
            },
            headers={"X-Request-ID": request_id, "Content-Type": "application/json"},
        ),
        FakeResponse(
            200,
            headers={"X-Request-ID": request_id, "Content-Type": "text/html; charset=utf-8"},
            text="<title>Data Prism</title>",
        ),
    ]


class ReleaseCheckTests(unittest.TestCase):
    def test_public_target_requires_https_and_no_path(self):
        self.assertEqual(
            normalize_base_url("https://data-prism.onrender.com/"),
            "https://data-prism.onrender.com",
        )
        self.assertEqual(normalize_base_url("http://127.0.0.1:5001"), "http://127.0.0.1:5001")
        for target in (
            "http://example.com",
            "https://user:secret@example.com",
            "https://example.com/vibedash",
            "https://example.com?token=secret",
        ):
            with self.subTest(target=target):
                with self.assertRaises(ReleaseCheckInputError):
                    normalize_base_url(target)

    def test_short_and_full_revisions_match(self):
        self.assertTrue(revisions_match("e1d2bae123456789", "e1d2bae12345"))
        self.assertFalse(revisions_match("e1d2bae123456789", "3de358c4b30c"))
        self.assertFalse(revisions_match("main", "e1d2bae12345"))

    def test_successful_demo_release_reports_ephemeral_boundary(self):
        request_id = "release-check-test"
        session = FakeSession(successful_responses(request_id))

        report = verify_release(
            "https://data-prism.onrender.com",
            "e1d2bae123456789",
            session=session,
            request_id=request_id,
            checked_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["observed_revision"], "e1d2bae12345")
        self.assertEqual(len(report["checks"]), 6)
        self.assertTrue(all(item["status"] == "passed" for item in report["checks"]))
        self.assertIn("ephemeral", " ".join(report["warnings"]).lower())
        self.assertIn("real-user value is not proven", render_human_report(report))
        self.assertEqual(
            [call[0] for call in session.calls],
            [
                "https://data-prism.onrender.com/healthz",
                "https://data-prism.onrender.com/readyz",
                "https://data-prism.onrender.com/vibedash/",
            ],
        )
        self.assertTrue(all(call[1]["allow_redirects"] is False for call in session.calls))

    def test_revision_mismatch_fails_closed(self):
        report = verify_release(
            "https://data-prism.onrender.com",
            "aaaaaaaaaaaa",
            session=FakeSession(successful_responses()),
            request_id="release-check-test",
        )

        revision = next(
            item for item in report["checks"] if item["check_id"] == "deployed-revision"
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(revision["status"], "failed")
        self.assertIn("do not start a pilot", render_human_report(report))

    def test_not_ready_response_and_version_difference_fail(self):
        responses = successful_responses()
        responses[1] = FakeResponse(
            503,
            {
                "status": "not_ready",
                "issues": ["account store is unavailable."],
                "warnings": [],
                "version": "different123",
            },
            headers={"X-Request-ID": "release-check-test"},
        )

        report = verify_release(
            "https://data-prism.onrender.com",
            "e1d2bae123456789",
            session=FakeSession(responses),
            request_id="release-check-test",
        )

        failures = {
            item["check_id"]
            for item in report["checks"]
            if item["status"] == "failed"
        }
        self.assertIn("readiness-endpoint", failures)
        self.assertIn("endpoint-version-consistency", failures)

    def test_request_id_and_landing_contract_fail_closed(self):
        responses = successful_responses()
        responses[0].headers["X-Request-ID"] = "server-owned-id"
        responses[2] = FakeResponse(
            200,
            headers={"Content-Type": "text/plain"},
            text="not the application",
        )

        report = verify_release(
            "https://data-prism.onrender.com",
            "e1d2bae123456789",
            session=FakeSession(responses),
            request_id="release-check-test",
        )

        failures = {
            item["check_id"]
            for item in report["checks"]
            if item["status"] == "failed"
        }
        self.assertEqual(report["status"], "failed")
        self.assertIn("request-correlation", failures)
        self.assertIn("vibedash-landing", failures)

    def test_network_errors_are_reported_without_a_traceback(self):
        report = verify_release(
            "https://data-prism.onrender.com",
            "e1d2bae123456789",
            session=FakeSession(error=requests.ConnectionError("private detail")),
            request_id="release-check-test",
        )

        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["observed_revision"])
        self.assertTrue(all(item["status"] == "failed" for item in report["checks"]))
        self.assertNotIn("private detail", render_human_report(report))

    def test_invalid_timeout_is_rejected(self):
        with self.assertRaises(ReleaseCheckInputError):
            verify_release(
                "https://data-prism.onrender.com",
                "e1d2bae123456789",
                timeout=0,
                session=FakeSession(),
            )


if __name__ == "__main__":
    unittest.main()
