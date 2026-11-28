"""Choose which rows to analyse when a table exceeds the row limit.

Keeping the first rows is not neutral: in a log sorted by date it keeps only
the oldest period, and in an export sorted by customer or region it keeps only
the first few groups. Dated tables keep their most recent rows, so trends and
"what changed recently" describe the present; only the oldest kept period can
be partial. Other tables keep a seeded random sample, which leaves averages,
shares and relationships unbiased (totals then cover the sample only).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.column_semantics import columns_for_role, profile_dataframe_columns


RANDOM_STATE = 42


def select_rows(
    frame: pd.DataFrame,
    limit: int,
    *,
    column_profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return at most ``limit`` rows in their original order and how they were chosen."""
    total = int(len(frame))
    if total <= limit:
        return frame, {"method": "all_rows", "source_rows": total, "analyzed_rows": total}
    time_column, timestamps = _best_time_column(frame, column_profiles)
    if time_column is not None:
        # Undated rows sort first, so they are dropped before any dated row.
        order = np.argsort(timestamps.to_numpy(dtype="int64", na_value=np.iinfo("int64").min), kind="stable")
        positions = np.sort(order[-limit:])
        kept = timestamps.iloc[positions]
        info = {
            "method": "latest_rows_by_time",
            "time_column": str(time_column),
            "kept_from": kept.min().isoformat(),
            "kept_to": kept.max().isoformat(),
        }
    else:
        positions = np.sort(
            np.random.default_rng(RANDOM_STATE).choice(total, size=limit, replace=False)
        )
        info = {"method": "random_sample", "seed": RANDOM_STATE}
    info.update({"source_rows": total, "analyzed_rows": int(limit)})
    return frame.iloc[positions].reset_index(drop=True), info


def _best_time_column(
    frame: pd.DataFrame,
    column_profiles: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Tuple[Optional[Any], Optional[pd.Series]]:
    profiles = column_profiles if column_profiles is not None else profile_dataframe_columns(frame)
    best, best_values, best_count = None, None, 0
    for column in columns_for_role(frame, "temporal", profiles=profiles):
        values = frame[column]
        if not pd.api.types.is_datetime64_any_dtype(values):
            values = pd.to_datetime(values, errors="coerce", format="mixed")
        if getattr(values.dt, "tz", None) is not None:
            values = values.dt.tz_convert("UTC").dt.tz_localize(None)
        count = int(values.notna().sum())
        if count > best_count:
            best, best_values, best_count = column, values, count
    if best is None or best_count < len(frame) * 0.5:
        return None, None
    return best, best_values
