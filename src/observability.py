"""Small, dependency-free observability layer for the Flask runtime."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from flask import Flask, g, has_request_context, request


REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_STRUCTURED_FIELDS = (
    "event",
    "request_id",
    "http_method",
    "http_route",
    "http_status",
    "duration_ms",
    "service_version",
)


def service_version() -> str:
    """Return a short deployment identifier without invoking Git at runtime."""
    configured = os.getenv("SERVICE_VERSION") or os.getenv("RENDER_GIT_COMMIT")
    if not configured:
        return "development"
    return configured.strip()[:12] or "development"


def normalize_request_id(candidate: str | None) -> str:
    """Accept a bounded trace identifier or generate a server-owned one."""
    if candidate and _REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


class JsonLogFormatter(logging.Formatter):
    """Serialize application logs as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class RequestContextFilter(logging.Filter):
    """Attach deployment and request context to application events when available."""

    def __init__(self, version: str) -> None:
        super().__init__()
        self.version = version

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "service_version", None) is None:
            record.service_version = self.version
        if has_request_context() and getattr(record, "request_id", None) is None:
            record.request_id = g.get("request_id")
        return True


def configure_observability(app: Flask) -> None:
    """Configure application logs and safe per-request correlation metadata."""
    version = service_version()
    app.config["SERVICE_VERSION"] = version

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    app.logger.setLevel(level)

    if os.getenv("LOG_FORMAT", "text").lower() == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.addFilter(RequestContextFilter(version))
        handler.setFormatter(JsonLogFormatter())
        app.logger.handlers.clear()
        app.logger.addHandler(handler)
        app.logger.propagate = False

    @app.before_request
    def start_request_observation() -> None:
        g.request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        g.request_started_at = time.perf_counter()

    @app.after_request
    def finish_request_observation(response):
        duration_ms = round(
            (time.perf_counter() - g.get("request_started_at", time.perf_counter()))
            * 1000,
            2,
        )
        route = request.url_rule.rule if request.url_rule else "unmatched"
        request_id = g.get("request_id", uuid.uuid4().hex)

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "same-origin")

        app.logger.info(
            "HTTP %s %s %s in %.2fms",
            request.method,
            route,
            response.status_code,
            duration_ms,
            extra={
                "event": "http_request",
                "request_id": request_id,
                "http_method": request.method,
                "http_route": route,
                "http_status": response.status_code,
                "duration_ms": duration_ms,
                "service_version": version,
            },
        )
        return response
