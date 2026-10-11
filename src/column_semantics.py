"""Shared, data-aware column roles for every Data Prism analytical engine.

The contract deliberately separates storage dtype from analytical meaning.
Integer category codes are categories, sequential row keys are identifiers,
and parseable dates are temporal fields even when pandas stores them as text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


ColumnProfile = Dict[str, Any]
ColumnProfiles = Dict[Any, ColumnProfile]

_CURRENCY_TERMS = {
    "amount", "cost", "expense", "income", "margin", "price", "profit",
    "revenue", "sales", "spend",
}
_PERCENT_TERMS = {"percentage", "percent", "pct", "rate", "ratio", "share"}
_ADDITIVE_TERMS = {
    "amount", "cnt", "count", "expense", "income", "quantity", "revenue",
    "sales", "spend", "total", "units", "volume",
}
_MEASURE_TERMS = _CURRENCY_TERMS | _PERCENT_TERMS | _ADDITIVE_TERMS | {
    "age", "atemp", "distance", "duration", "hum", "humidity", "score",
    "feature", "measurement", "metric", "speed", "temp", "temperature",
    "value", "weight", "windspeed", "x", "y",
}
_TEMPORAL_TERMS = {
    "date", "datetime", "day", "month", "quarter", "time", "timestamp",
    "week", "year",
}
_IDENTIFIER_TERMS = {"guid", "id", "identifier", "key", "uuid"}
_CATEGORY_TERMS = {
    "category", "channel", "class", "cohort", "country", "group", "holiday",
    "mnth", "plan", "region", "season", "segment", "status", "type",
    "weekday", "weathersit", "workingday", "yr",
}


def column_terms(column: Any) -> set[str]:
    """Return normalized name tokens, including camel-case boundaries."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(column))
    return {
        token
        for token in re.split(r"[^a-zA-Z0-9а-яА-ЯёЁ]+", name.lower())
        if token
    }


def is_temporal_name(column: Any) -> bool:
    return bool(column_terms(column) & _TEMPORAL_TERMS)


def is_identifier_name(column: Any) -> bool:
    """Recognize names that identify rows or entities rather than measures."""
    terms = column_terms(column)
    normalized = re.sub(r"[^a-z0-9]+", "", str(column).lower())
    return (
        normalized
        in {
            "index", "rank", "ranking", "recordid", "rk", "rowid",
            "rowindex", "rownumber",
        }
        or bool(terms & (_IDENTIFIER_TERMS - {"id", "key"}))
        or "id" in terms
        or ("key" in terms and str(column).lower().rstrip().endswith("key"))
    )


def profile_dataframe_columns(
    dataframe: Optional[pd.DataFrame],
    columns: Optional[Iterable[Any]] = None,
) -> ColumnProfiles:
    """Build bounded aggregate profiles and assign one analytical role."""
    if dataframe is None:
        return {}

    selected = list(dataframe.columns if columns is None else columns)
    profiles: ColumnProfiles = {}
    for column in selected:
        if column not in dataframe.columns:
            continue
        series = dataframe[column]
        non_missing_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        is_numeric = bool(is_numeric_dtype(series.dtype)) and not bool(
            pd.api.types.is_complex_dtype(series.dtype)
        )
        is_temporal = bool(is_datetime64_any_dtype(series.dtype))
        if not is_temporal and not is_numeric and non_missing_count:
            sample = series.dropna().astype(str).head(200)
            date_like = sample.str.match(
                r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
                r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4})(?:[T\s].*)?\s*$"
            )
            if not sample.empty and date_like.mean() >= 0.9:
                parsed = pd.to_datetime(sample, errors="coerce")
                is_temporal = bool(parsed.notna().mean() >= 0.9)

        numeric_values = (
            pd.to_numeric(series.dropna(), errors="coerce") if is_numeric else None
        )
        is_integer_like = bool(
            is_numeric
            and numeric_values is not None
            and not numeric_values.empty
            and ((numeric_values - numeric_values.round()).abs() <= 1e-9).all()
        )
        is_monotonic_unique = bool(
            is_integer_like
            and non_missing_count >= 3
            and unique_count == non_missing_count
            and (
                numeric_values.is_monotonic_increasing
                or numeric_values.is_monotonic_decreasing
            )
        )
        is_sequential_identifier = bool(
            is_monotonic_unique
            and numeric_values.diff().dropna().abs().eq(1).all()
        )
        low_cardinality_integer = bool(
            is_integer_like
            and unique_count > 1
            and non_missing_count > 0
            and (unique_count <= 20 or unique_count / non_missing_count <= 0.02)
        )
        terms = column_terms(column)

        if is_temporal or is_temporal_name(column):
            role = "temporal"
        elif is_identifier_name(column):
            role = "identifier"
        elif is_numeric and terms & _MEASURE_TERMS:
            role = "measure"
        elif terms & _CATEGORY_TERMS:
            role = "category"
        elif is_sequential_identifier:
            role = "identifier"
        elif is_numeric and low_cardinality_integer:
            role = "category"
        elif is_numeric:
            role = "measure"
        else:
            role = "category"

        finite_count = 0
        if is_numeric:
            numeric_series = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            finite_count = int(numeric_series.notna().sum())
        profiles[column] = {
            "role": role,
            "is_numeric": is_numeric,
            "is_temporal": is_temporal,
            "is_integer_like": is_integer_like,
            "is_monotonic_unique": is_monotonic_unique,
            "is_sequential_identifier": is_sequential_identifier,
            "non_missing_count": non_missing_count,
            "finite_count": finite_count,
            "unique_count": unique_count,
            "unique_ratio": unique_count / non_missing_count if non_missing_count else 0.0,
        }
    return profiles


def column_role(
    column: Any,
    profiles: Mapping[Any, Mapping[str, Any]],
) -> Optional[str]:
    role = profiles.get(column, {}).get("role")
    return str(role) if role in {"measure", "category", "temporal", "identifier"} else None


def columns_for_role(
    dataframe: pd.DataFrame,
    role: str,
    *,
    profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    require_variation: bool = False,
) -> list[Any]:
    if role not in {"measure", "category", "temporal", "identifier"}:
        raise ValueError(f"Unsupported analytical role: {role}")
    resolved = profiles or profile_dataframe_columns(dataframe)
    columns = []
    for column in dataframe.columns:
        profile = resolved.get(column, {})
        if profile.get("role") != role:
            continue
        if require_variation and int(profile.get("unique_count", 0)) < 2:
            continue
        columns.append(column)
    return columns
