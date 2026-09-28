"""Validated, deterministic measurements for decision outcomes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any


OUTCOME_MEASUREMENT_CONTRACT = "decision-outcome-measurement-v1"
_MAX_NUMBER_CHARACTERS = 40
_MAX_ABSOLUTE_VALUE = Decimal("1e18")
_MAX_DECIMAL_PLACES = 8
_DECIMAL_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{label} is required.")
    if len(normalized) > maximum:
        raise ValueError(f"{label} must not exceed {maximum} characters.")
    return normalized


def _number(value: Any, label: str) -> tuple[Decimal, str]:
    text = _text(value, label, maximum=_MAX_NUMBER_CHARACTERS)
    if not _DECIMAL_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must be a plain decimal number.")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{label} must be a decimal number.") from error
    if not number.is_finite():
        raise ValueError(f"{label} must be finite.")
    if abs(number) > _MAX_ABSOLUTE_VALUE:
        raise ValueError(f"{label} is outside the supported range.")
    decimal_places = max(0, -number.as_tuple().exponent)
    if decimal_places > _MAX_DECIMAL_PLACES:
        raise ValueError(
            f"{label} must use at most {_MAX_DECIMAL_PLACES} decimal places."
        )
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"-0", ""}:
        normalized = "0"
    return number, normalized


def _date(value: Any, label: str) -> str:
    text = _text(value, label, maximum=10)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must use YYYY-MM-DD format.") from error
    return parsed.isoformat()


def build_outcome_measurement(
    *,
    baseline_value: Any = "",
    observed_value: Any = "",
    unit: Any = "",
    period_start: Any = "",
    period_end: Any = "",
    required: bool = False,
) -> dict[str, Any]:
    """Return a complete measurement or an empty optional measurement."""
    if not isinstance(required, bool):
        raise ValueError("Measurement requirement must be boolean.")
    values = (baseline_value, observed_value, unit, period_start, period_end)
    empty = [isinstance(value, str) and not value.strip() for value in values]
    if all(empty):
        if required:
            raise ValueError(
                "Validated and invalidated outcomes require baseline, observed "
                "value, unit, and observation period."
            )
        return {}
    if any(empty):
        raise ValueError(
            "Complete every outcome measurement field or leave all of them blank."
        )

    baseline, baseline_text = _number(baseline_value, "Baseline value")
    observed, observed_text = _number(observed_value, "Observed value")
    normalized_unit = _text(unit, "Measurement unit", maximum=40)
    start = _date(period_start, "Observation start")
    end = _date(period_end, "Observation end")
    if start > end:
        raise ValueError("Observation start must not be after observation end.")

    delta = observed - baseline
    delta_text = format(delta, "f")
    if "." in delta_text:
        delta_text = delta_text.rstrip("0").rstrip(".")
    if delta_text == "-0":
        delta_text = "0"
    relative_change = None
    if baseline != 0:
        with localcontext() as context:
            context.prec = 64
            relative_change = float(
                (
                    (observed - baseline) / abs(baseline) * Decimal("100")
                ).quantize(Decimal("0.01"))
            )

    return {
        "contract": OUTCOME_MEASUREMENT_CONTRACT,
        "baseline_value": baseline_text,
        "observed_value": observed_text,
        "unit": normalized_unit,
        "period_start": start,
        "period_end": end,
        "delta_value": delta_text,
        "relative_change_percent": relative_change,
    }


def normalize_outcome_measurement(
    value: Mapping[str, Any] | None,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Validate a retained measurement and recompute all derived values."""
    if value in (None, {}):
        return build_outcome_measurement(required=required)
    if not isinstance(value, Mapping):
        raise ValueError("Outcome measurement is invalid.")
    if value.get("contract") != OUTCOME_MEASUREMENT_CONTRACT:
        raise ValueError("Outcome measurement contract is unsupported.")
    return build_outcome_measurement(
        baseline_value=value.get("baseline_value"),
        observed_value=value.get("observed_value"),
        unit=value.get("unit"),
        period_start=value.get("period_start"),
        period_end=value.get("period_end"),
        required=required,
    )


def serialize_outcome_measurement(value: Mapping[str, Any] | None) -> str:
    """Serialize only validated base fields; derived values are recomputed."""
    normalized = normalize_outcome_measurement(value)
    if not normalized:
        return "{}"
    retained = {
        key: normalized[key]
        for key in (
            "contract",
            "baseline_value",
            "observed_value",
            "unit",
            "period_start",
            "period_end",
        )
    }
    return json.dumps(retained, sort_keys=True, separators=(",", ":"))


def deserialize_outcome_measurement(value: Any) -> dict[str, Any]:
    """Load and validate a retained measurement JSON value."""
    if not isinstance(value, str):
        raise ValueError("Outcome measurement storage is invalid.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("Outcome measurement storage is invalid.") from error
    return normalize_outcome_measurement(parsed)
