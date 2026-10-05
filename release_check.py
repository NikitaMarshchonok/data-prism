"""Verify that the intended Data Prism revision is serving public pilot traffic.

This command is deliberately read-only. It checks the public liveness,
readiness, revision, request-correlation, and VibeDash landing contracts without
creating an account, uploading data, or starting an analysis.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


CONTRACT = "data-prism-release-check-v1"
DEFAULT_BASE_URL = "https://data-prism.onrender.com"
REQUEST_ID_HEADER = "X-Request-ID"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{7,40}")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ReleaseCheckInputError(ValueError):
    """Raised when the operator supplies an unsafe or ambiguous target."""


def normalize_base_url(value: str) -> str:
    """Return a canonical origin and reject credentials or non-local HTTP."""
    parsed = urlsplit((value or "").strip())
    if not parsed.scheme or not parsed.hostname:
        raise ReleaseCheckInputError("Base URL must include a scheme and host.")
    if parsed.username or parsed.password:
        raise ReleaseCheckInputError("Base URL must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise ReleaseCheckInputError("Base URL must not contain a query or fragment.")
    if parsed.path not in {"", "/"}:
        raise ReleaseCheckInputError("Base URL must not contain an application path.")
    if parsed.scheme not in {"http", "https"}:
        raise ReleaseCheckInputError("Base URL must use HTTP or HTTPS.")
    if parsed.scheme != "https" and parsed.hostname not in _LOCAL_HOSTS:
        raise ReleaseCheckInputError("Public release checks require HTTPS.")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def normalize_revision(value: str) -> str:
    """Validate a Git revision without accepting branch names or arbitrary text."""
    revision = (value or "").strip().lower()
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ReleaseCheckInputError(
            "Expected revision must be a 7- to 40-character hexadecimal Git SHA."
        )
    return revision


def revisions_match(expected: str, observed: str) -> bool:
    """Compare full and shortened Git SHAs while keeping a seven-char minimum."""
    try:
        expected_revision = normalize_revision(expected)
        observed_revision = normalize_revision(observed)
    except ReleaseCheckInputError:
        return False
    return expected_revision.startswith(observed_revision) or observed_revision.startswith(
        expected_revision
    )


def _check(check_id: str, passed: bool, detail: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "status": "passed" if passed else "failed",
        "detail": detail,
    }


def _request(
    session: requests.Session,
    url: str,
    *,
    request_id: str,
    timeout: float,
) -> tuple[requests.Response | None, str | None]:
    try:
        response = session.get(
            url,
            headers={REQUEST_ID_HEADER: request_id},
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.RequestException as error:
        return None, f"{error.__class__.__name__}: request failed"
    return response, None


def _json_object(response: requests.Response) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None, "response is not valid JSON"
    if not isinstance(payload, dict):
        return None, "response JSON is not an object"
    return payload, None


def verify_release(
    base_url: str,
    expected_revision: str,
    *,
    timeout: float = 90.0,
    session: requests.Session | None = None,
    request_id: str | None = None,
    checked_at: datetime | None = None,
) -> dict[str, Any]:
    """Run a read-only public release check and return a stable JSON contract."""
    target = normalize_base_url(base_url)
    expected = normalize_revision(expected_revision)
    if timeout <= 0 or timeout > 300:
        raise ReleaseCheckInputError("Timeout must be greater than 0 and at most 300 seconds.")

    correlation_id = request_id or f"release-check-{secrets.token_hex(8)}"
    client = session or requests.Session()
    checks: list[dict[str, str]] = []
    warnings: list[str] = []
    observed_revision: str | None = None
    health_payload: dict[str, Any] | None = None
    ready_payload: dict[str, Any] | None = None

    health_response, health_error = _request(
        client,
        f"{target}/healthz",
        request_id=correlation_id,
        timeout=timeout,
    )
    if health_response is None:
        checks.append(_check("health-endpoint", False, health_error or "request failed"))
    else:
        health_payload, json_error = _json_object(health_response)
        health_ok = bool(
            health_response.status_code == 200
            and not json_error
            and health_payload
            and health_payload.get("status") == "ok"
            and health_payload.get("service") == "data-prism"
        )
        checks.append(
            _check(
                "health-endpoint",
                health_ok,
                "HTTP 200 and the Data Prism liveness contract are present"
                if health_ok
                else f"HTTP {health_response.status_code}; {json_error or 'contract mismatch'}",
            )
        )
        if health_payload:
            candidate = health_payload.get("version")
            observed_revision = candidate if isinstance(candidate, str) else None
        request_id_ok = health_response.headers.get(REQUEST_ID_HEADER) == correlation_id
        checks.append(
            _check(
                "request-correlation",
                request_id_ok,
                "the supplied request ID was preserved"
                if request_id_ok
                else "the supplied request ID was not preserved",
            )
        )

    revision_ok = bool(
        observed_revision and revisions_match(expected, observed_revision)
    )
    checks.append(
        _check(
            "deployed-revision",
            revision_ok,
            f"expected {expected}; observed {observed_revision or 'unavailable'}",
        )
    )

    ready_response, ready_error = _request(
        client,
        f"{target}/readyz",
        request_id=correlation_id,
        timeout=timeout,
    )
    if ready_response is None:
        checks.append(_check("readiness-endpoint", False, ready_error or "request failed"))
    else:
        ready_payload, json_error = _json_object(ready_response)
        ready_ok = bool(
            ready_response.status_code == 200
            and not json_error
            and ready_payload
            and ready_payload.get("status") == "ready"
            and ready_payload.get("issues") == []
        )
        checks.append(
            _check(
                "readiness-endpoint",
                ready_ok,
                "HTTP 200 with no readiness issues"
                if ready_ok
                else f"HTTP {ready_response.status_code}; {json_error or 'not ready'}",
            )
        )
        ready_revision = ready_payload.get("version") if ready_payload else None
        version_consistent = bool(
            isinstance(ready_revision, str)
            and observed_revision
            and ready_revision == observed_revision
        )
        checks.append(
            _check(
                "endpoint-version-consistency",
                version_consistent,
                "health and readiness report the same revision"
                if version_consistent
                else "health and readiness revisions differ or are unavailable",
            )
        )
        if ready_payload:
            payload_warnings = ready_payload.get("warnings")
            if isinstance(payload_warnings, list):
                warnings.extend(str(item) for item in payload_warnings if item)

    landing_response, landing_error = _request(
        client,
        f"{target}/vibedash/",
        request_id=correlation_id,
        timeout=timeout,
    )
    if landing_response is None:
        checks.append(_check("vibedash-landing", False, landing_error or "request failed"))
    else:
        content_type = landing_response.headers.get("Content-Type", "").lower()
        landing_ok = bool(
            landing_response.status_code == 200
            and "text/html" in content_type
            and "Data Prism" in landing_response.text
        )
        checks.append(
            _check(
                "vibedash-landing",
                landing_ok,
                "VibeDash landing page is available"
                if landing_ok
                else f"HTTP {landing_response.status_code}; landing contract mismatch",
            )
        )

    runtime_storage = ready_payload.get("runtime_storage") if ready_payload else None
    if isinstance(runtime_storage, dict) and runtime_storage.get("state_durability") == "ephemeral":
        warnings.append(
            "Runtime state is ephemeral; this pass supports only the documented "
            "supervised pilot boundary."
        )

    unique_warnings = list(dict.fromkeys(warnings))
    passed = all(item["status"] == "passed" for item in checks)
    timestamp = checked_at or datetime.now(timezone.utc)
    return {
        "contract": CONTRACT,
        "status": "passed" if passed else "failed",
        "checked_at": timestamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "base_url": target,
        "expected_revision": expected,
        "observed_revision": observed_revision,
        "checks": checks,
        "warnings": unique_warnings,
        "runtime_storage": runtime_storage if isinstance(runtime_storage, dict) else None,
    }


def render_human_report(report: dict[str, Any]) -> str:
    """Render the verification result without overstating pilot suitability."""
    lines = [
        f"Data Prism release check: {report['status'].upper()}",
        f"Target: {report['base_url']}",
        f"Revision: expected {report['expected_revision']}; "
        f"observed {report.get('observed_revision') or 'unavailable'}",
    ]
    for item in report["checks"]:
        marker = "PASS" if item["status"] == "passed" else "FAIL"
        lines.append(f"[{marker}] {item['check_id']}: {item['detail']}")
    for warning in report.get("warnings", []):
        lines.append(f"[WARN] {warning}")
    if report["status"] == "passed":
        lines.append("Result: technical release checks passed; real-user value is not proven.")
    else:
        lines.append("Result: do not start a pilot session until failed checks are resolved.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the deployed Data Prism revision before a pilot session."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the stable JSON contract instead of the human report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = verify_release(
            args.base_url,
            args.expected_revision,
            timeout=args.timeout,
        )
    except ReleaseCheckInputError as error:
        parser.error(str(error))
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_human_report(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
