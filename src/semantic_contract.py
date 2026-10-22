"""Validated, user-confirmed semantics for one tabular analysis.

The contract contains schema metadata only.  It never includes source row
values and is safe to retain beside an analysis audit manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.column_semantics import column_terms, profile_dataframe_columns


SEMANTIC_CONTRACT_VERSION = "confirmed-semantic-contract-v1"
MAX_CONTRACT_COLUMNS = 200
MAX_CONTRACT_JSON_BYTES = 100_000
# A normal contract needs only object -> columns array -> column object.
# Keep a fixed budget instead of relying on Python's JSON recursion behavior.
MAX_CONTRACT_JSON_DEPTH = 8

ALLOWED_ROLES = {"measure", "category", "temporal", "identifier"}
ALLOWED_UNITS = {"none", "number", "count", "currency", "ratio", "percent"}

_CURRENCY_TERMS = {
    "amount", "cost", "expense", "income", "margin", "price", "profit",
    "revenue", "sales", "spend",
}
_PERCENT_TERMS = {"percentage", "percent", "pct", "rate", "ratio", "share"}
_COUNT_TERMS = {"cnt", "count", "quantity", "records", "units", "volume"}


class SemanticContractError(ValueError):
    """Raised when submitted semantics do not match the analyzed dataset."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inferred_unit(
    dataframe: pd.DataFrame,
    column: Any,
    profile: Mapping[str, Any],
) -> str:
    if profile.get("role") != "measure" or not profile.get("is_numeric"):
        return "none"
    terms = column_terms(column)
    if terms & _CURRENCY_TERMS:
        return "currency"
    if terms & _PERCENT_TERMS:
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty or float(values.abs().max()) <= 1.5:
            return "ratio"
        return "percent"
    if terms & _COUNT_TERMS and profile.get("is_integer_like"):
        return "count"
    return "number"


def infer_semantic_contract(
    dataframe: pd.DataFrame,
    *,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Build a bounded proposal that a user must review before analysis."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if len(dataframe.columns) > MAX_CONTRACT_COLUMNS:
        raise SemanticContractError(
            f"Semantic confirmation supports at most {MAX_CONTRACT_COLUMNS} columns."
        )

    profiles = profile_dataframe_columns(dataframe)
    time_selected = False
    columns = []
    for column in dataframe.columns:
        profile = profiles[column]
        role = str(profile["role"])
        is_time = role == "temporal" and not time_selected
        time_selected = time_selected or is_time
        columns.append({
            "name": str(column),
            "role": role,
            "unit": _inferred_unit(dataframe, column, profile),
            "is_target": False,
            "is_time": is_time,
            "exclude_from_predictors": role in {"identifier", "temporal"},
        })
    proposal = {
        "contract": SEMANTIC_CONTRACT_VERSION,
        "confirmed": bool(confirmed),
        "columns": columns,
    }
    return validate_semantic_contract(
        dataframe,
        proposal,
        require_confirmation=confirmed,
    )


def parse_semantic_contract_json(value: Any) -> Mapping[str, Any]:
    """Parse a bounded form field without accepting non-object JSON."""
    if not isinstance(value, str) or not value.strip():
        raise SemanticContractError(
            "Review and confirm the dataset semantics before analysis."
        )
    if len(value.encode("utf-8")) > MAX_CONTRACT_JSON_BYTES:
        raise SemanticContractError("The semantic contract is too large.")
    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > MAX_CONTRACT_JSON_DEPTH:
                raise SemanticContractError("The semantic contract is nested too deeply.")
        elif character in "}]":
            depth -= 1
            if depth < 0:
                raise SemanticContractError("The semantic contract is not valid JSON.")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError) as error:
        raise SemanticContractError("The semantic contract is not valid JSON.") from error
    if not isinstance(parsed, Mapping):
        raise SemanticContractError("The semantic contract must be a JSON object.")
    return parsed


def validate_semantic_contract(
    dataframe: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    require_confirmation: bool = True,
) -> dict[str, Any]:
    """Validate and normalize one contract against the actual dataframe."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(contract, Mapping):
        raise SemanticContractError("The semantic contract must be an object.")
    if contract.get("contract") != SEMANTIC_CONTRACT_VERSION:
        raise SemanticContractError("The semantic contract version is unsupported.")
    if require_confirmation and contract.get("confirmed") is not True:
        raise SemanticContractError(
            "Review and confirm the dataset semantics before analysis."
        )

    dataset_columns = [str(column) for column in dataframe.columns]
    if len(dataset_columns) > MAX_CONTRACT_COLUMNS:
        raise SemanticContractError(
            f"Semantic confirmation supports at most {MAX_CONTRACT_COLUMNS} columns."
        )
    if len(set(dataset_columns)) != len(dataset_columns):
        raise SemanticContractError("Dataset column names must be unique.")

    submitted_columns = contract.get("columns")
    if not isinstance(submitted_columns, list):
        raise SemanticContractError("The semantic contract columns are missing.")
    submitted_names = [
        item.get("name") if isinstance(item, Mapping) else None
        for item in submitted_columns
    ]
    if submitted_names != dataset_columns:
        raise SemanticContractError(
            "The semantic contract no longer matches the uploaded dataset."
        )

    inferred = profile_dataframe_columns(dataframe)
    normalized_columns = []
    target_count = 0
    time_count = 0
    for original_column, submitted in zip(dataframe.columns, submitted_columns):
        if not isinstance(submitted, Mapping):
            raise SemanticContractError("Each column semantic must be an object.")
        name = str(original_column)
        role = submitted.get("role")
        unit = submitted.get("unit")
        is_target = submitted.get("is_target")
        is_time = submitted.get("is_time")
        excluded = submitted.get("exclude_from_predictors")
        if not isinstance(role, str) or role not in ALLOWED_ROLES:
            raise SemanticContractError(f"Column '{name}' has an unsupported role.")
        if not isinstance(unit, str) or unit not in ALLOWED_UNITS:
            raise SemanticContractError(f"Column '{name}' has an unsupported unit.")
        if not all(isinstance(flag, bool) for flag in (is_target, is_time, excluded)):
            raise SemanticContractError(f"Column '{name}' has invalid semantic flags.")

        profile = inferred[original_column]
        is_numeric = bool(profile.get("is_numeric"))
        if role == "measure" and not is_numeric:
            raise SemanticContractError(
                f"Column '{name}' cannot be a measure because it is not numeric."
            )
        if unit != "none" and (role != "measure" or not is_numeric):
            raise SemanticContractError(
                f"Column '{name}' can only have a numeric unit when it is a measure."
            )
        if role == "measure" and unit == "none":
            raise SemanticContractError(f"Measure '{name}' requires a unit.")
        if unit == "count" and not profile.get("is_integer_like"):
            raise SemanticContractError(
                f"Column '{name}' cannot use the count unit because values are not integers."
            )
        if role == "temporal" and profile.get("role") != "temporal":
            raise SemanticContractError(
                f"Column '{name}' is not compatible with the temporal role."
            )
        if is_target and role not in {"measure", "category"}:
            raise SemanticContractError(
                f"Column '{name}' cannot be the target with role '{role}'."
            )
        if is_time and role != "temporal":
            raise SemanticContractError(
                f"Column '{name}' can only be the time axis with the temporal role."
            )
        if is_target and is_time:
            raise SemanticContractError(
                f"Column '{name}' cannot be both the target and time axis."
            )
        if (is_target or role == "identifier") and not excluded:
            raise SemanticContractError(
                f"Column '{name}' must be excluded from predictive features."
            )
        if (
            role == "category"
            and float(profile.get("unique_ratio", 0.0)) >= 0.95
            and int(profile.get("non_missing_count", 0)) >= 20
            and not excluded
        ):
            raise SemanticContractError(
                f"Near-unique category '{name}' must be excluded from predictive features."
            )

        target_count += int(is_target)
        time_count += int(is_time)
        normalized_columns.append({
            "name": name,
            "role": role,
            "unit": unit,
            "is_target": is_target,
            "is_time": is_time,
            "exclude_from_predictors": excluded,
        })

    if target_count > 1:
        raise SemanticContractError("Select at most one target column.")
    if time_count > 1:
        raise SemanticContractError("Select at most one time column.")

    normalized = {
        "contract": SEMANTIC_CONTRACT_VERSION,
        "confirmed": bool(contract.get("confirmed")),
        "columns": normalized_columns,
    }
    normalized["sha256"] = _canonical_sha256(normalized)
    return normalized


def profiles_from_semantic_contract(
    dataframe: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    prevalidated: bool = False,
) -> dict[Any, dict[str, Any]]:
    """Return current aggregate profiles with confirmed semantic overrides."""
    if prevalidated:
        normalized = contract
        expected_names = [str(column) for column in dataframe.columns]
        submitted = contract.get("columns") if isinstance(contract, Mapping) else None
        submitted_names = (
            [item.get("name") for item in submitted if isinstance(item, Mapping)]
            if isinstance(submitted, list)
            else []
        )
        if (
            contract.get("contract") != SEMANTIC_CONTRACT_VERSION
            or contract.get("confirmed") is not True
            or submitted_names != expected_names
        ):
            raise SemanticContractError(
                "The semantic contract no longer matches the analyzed dataset."
            )
    else:
        normalized = validate_semantic_contract(dataframe, contract)
    profiles = profile_dataframe_columns(dataframe)
    by_name = {item["name"]: item for item in normalized["columns"]}
    for column in dataframe.columns:
        semantic = by_name[str(column)]
        profiles[column] = {
            **profiles[column],
            "role": semantic["role"],
            "unit": semantic["unit"],
            "is_target": semantic["is_target"],
            "is_time": semantic["is_time"],
            "exclude_from_predictors": semantic["exclude_from_predictors"],
        }
    return profiles


def semantic_contract_audit_view(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded schema-only representation retained in manifests."""
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    if not isinstance(columns, list):
        raise SemanticContractError("The semantic contract columns are missing.")
    return {
        "contract": contract.get("contract"),
        "confirmed": contract.get("confirmed") is True,
        "sha256": contract.get("sha256"),
        "columns": [dict(item) for item in columns if isinstance(item, Mapping)],
    }
