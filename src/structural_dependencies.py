"""Conservative, bounded checks for observed exact integer sums.

This is not symbolic regression or a proof of business definitions. Only
complete, varying, safely representable integer measures are checked. A miss
does not establish independence or the absence of derived features.
"""

from itertools import combinations
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


MIN_ROWS = 30
MAX_ROWS = 100_000
MAX_COLUMNS = 20
MAX_DEPENDENCIES = 20
MAX_EXACT_OPERAND = 2 ** 52


def detect_additive_dependencies(
    df: pd.DataFrame,
    profiles: Mapping[Any, Mapping[str, Any]],
    context: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Verify total = component A + component B on every analyzed row.

    Focus columns precede schema order within the column cap. No missing-value
    imputation, fitting, rounding, or approximate matching is performed. Units
    must agree (including unknown with unknown); unit labels alone do not prove
    compatible business definitions. Large/fractional values are left unassessed.
    """
    if not MIN_ROWS <= len(df) <= MAX_ROWS or not df.columns.is_unique:
        return []
    context = context or {}
    excluded = set(context.get("excluded_columns") or [])
    focus = set(context.get("focus_columns") or [])
    columns = [column for column in df.columns
               if profiles.get(column, {}).get("role") == "measure"
               and str(column) not in excluded
               and pd.api.types.is_numeric_dtype(df[column])
               and not pd.api.types.is_bool_dtype(df[column])
               and not pd.api.types.is_complex_dtype(df[column])]
    columns.sort(key=lambda column: str(column) not in focus)
    values = {}
    for column in columns[:MAX_COLUMNS]:
        # Do not silently narrow extended precision into an integer equality.
        if pd.api.types.is_float_dtype(df[column]) and df[column].dtype.itemsize > 8:
            continue
        array = df[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if (not np.isfinite(array).all()
                or (np.abs(array) > MAX_EXACT_OPERAND).any()
                or not np.equal(array, np.trunc(array)).all()
                or np.unique(array).size < 3):
            continue
        values[column] = array

    # A cheap precheck only rejects candidates. Every accepted equality is
    # checked against ALL rows, including those outside this preview.
    preview = np.linspace(0, len(df) - 1, min(16, len(df)), dtype=int)
    dependencies = []
    for triple in combinations(values, 3):
        units = {profiles.get(column, {}).get("unit", "unknown") for column in triple}
        if len(units) != 1:
            continue
        for total in triple:
            components = [column for column in triple if column != total]
            a, b = (values[column] for column in components)
            target = values[total]
            if not np.equal(target[preview], a[preview] + b[preview]).all():
                continue
            if not np.equal(target, a + b).all():
                continue
            names = [str(column) for column in components]
            dependencies.append({
                "kind": "observed_integer_sum",
                "total_column": str(total),
                "component_columns": names,
                "columns": [str(total), *names],
                "equation": f"{total} = {names[0]} + {names[1]}",
                "evaluated_rows": int(len(df)),
                "coverage": 1.0,
                "comparison": "exact_safe_integer_sum",
            })
            if len(dependencies) == MAX_DEPENDENCIES:
                return dependencies
    return dependencies


def additive_pair_dependency(
    columns: list[str], dependencies: list[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Match only total/component pairs, not the two components themselves."""
    pair = set(columns)
    if len(pair) != 2:
        return None
    return next((item for item in dependencies
                 if item.get("total_column") in pair
                 and pair <= set(item.get("columns") or [])), None)
