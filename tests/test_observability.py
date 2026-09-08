import json
import logging
import unittest
from unittest.mock import patch

from flask import Flask

from src.observability import (
    JsonLogFormatter,
    configure_observability,
    normalize_request_id,
    service_version,
)


class ObservabilityTests(unittest.TestCase):
    def test_request_id_accepts_only_bounded_safe_values(self):
        self.assertEqual(normalize_request_id("trace-123:worker_1"), "trace-123:worker_1")
        self.assertRegex(normalize_request_id("contains spaces"), r"^[0-9a-f]{32}$")
        self.assertRegex(normalize_request_id("x" * 129), r"^[0-9a-f]{32}$")

    def test_render_commit_is_used_as_short_service_version(self):
        with patch.dict(
            "os.environ",
            {"RENDER_GIT_COMMIT": "1234567890abcdef", "SERVICE_VERSION": ""},
            clear=False,
        ):
            self.assertEqual(service_version(), "1234567890ab")

    def test_json_formatter_emits_structured_request_fields(self):
        record = logging.LogRecord(
            name="data-prism",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="HTTP request completed",
            args=(),
            exc_info=None,
        )
        record.event = "http_request"
        record.request_id = "trace-123"
        record.http_route = "/items/<item_id>"
        record.http_status = 200
        record.duration_ms = 4.25

        payload = json.loads(JsonLogFormatter().format(record))

        self.assertEqual(payload["event"], "http_request")
        self.assertEqual(payload["request_id"], "trace-123")
        self.assertEqual(payload["http_route"], "/items/<item_id>")
        self.assertEqual(payload["duration_ms"], 4.25)
        self.assertIn("timestamp", payload)

    def test_flask_responses_preserve_safe_request_id_and_add_headers(self):
        app = Flask(__name__)
        with patch.dict(
            "os.environ",
            {"LOG_FORMAT": "text", "SERVICE_VERSION": "test-release"},
            clear=False,
        ):
            configure_observability(app)

        @app.get("/items/<item_id>")
        def item(item_id):
            return {"item_id": item_id}

        with self.assertLogs(app.logger, level="INFO") as captured:
            with app.test_client() as client:
                response = client.get(
                    "/items/42?private=value",
                    headers={"X-Request-ID": "portfolio-check-001"},
                )

        self.assertEqual(response.headers["X-Request-ID"], "portfolio-check-001")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "SAMEORIGIN")
        self.assertEqual(response.headers["Referrer-Policy"], "same-origin")
        request_record = captured.records[-1]
        self.assertEqual(request_record.http_route, "/items/<item_id>")
        self.assertEqual(request_record.request_id, "portfolio-check-001")
        self.assertNotIn("private=value", request_record.getMessage())


if __name__ == "__main__":
    unittest.main()
