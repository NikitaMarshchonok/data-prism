"""Explain why a KPI changed: an exact decomposition of the latest change by segment.

The question a business owner asks first is "why did revenue drop?".  For a
dated table this engine compares the most recent complete window with a
comparable earlier one and splits the change across the categories in the
data:

* Summed measures (counts, currency): the total change is the exact sum of
  each segment's change, so "Europe accounts for 70% of the drop" is an
  accounting identity, not an estimate.
* Averaged measures (rates, prices, scores): the change in the average is
  split with a symmetric shift-share into a *mix* effect (the share of rows
  moved between segments that sit at different levels) and a *rate* effect
  (segments themselves moved).  This is how a middle data scientist catches
  Simpson's paradox: an average can fall while every segment improves.

Windows follow the measure's grain from the temporal analysis: the last 7 days
against the 7 before (daily), the last 4 complete weeks against the 4 before
(weekly) and the last complete month against the same month a year earlier
(monthly, which removes yearly seasonality).  Whether the change is larger
than usual is judged against the same comparison repeated over the history.

The decomposition describes where a change happened; it does not establish
why.  Everything is deterministic and no rows leave the process.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.column_semantics import columns_for_role, profile_dataframe_columns
from src.statistical_primitives import benjamini_hochberg

from .temporal_engine import DEFAULT_ALPHA, SUMMED_UNITS
from .temporal_patterns import robust_unusualness


DRIVER_CONTRACT = "driver-analysis-v1"
MAX_MEASURES = 3
MAX_DIMENSIONS = 5
MAX_SEGMENTS_PER_DIMENSION = 12
MAX_DIMENSION_CARDINALITY = 50
MIN_HISTORY_COMPARISONS = 6
MIN_ROWS_PER_WINDOW = 10
DISPLAYED_SEGMENTS = 3
OTHER_SEGMENT = "Other"
MISSING_SEGMENT = "(missing)"
#: Changes smaller than this are reported as flat and get no driver story.
MIN_MATERIAL_CHANGE = 0.05
#: A mix effect is named as the main story when it explains at least this share.
MIX_STORY_SHARE = 0.5
#: A segment "drives" a change when its share of the change exceeds its share
#: of the business by at least this much; below it the change is spread evenly.
MIN_DRIVER_EXCESS = 0.15
_WINDOWS = {
    "day": {"label": "the last 7 days against the 7 days before", "kind": "previous"},
    "week": {"label": "the last 4 complete weeks against the 4 weeks before", "kind": "previous"},
    "month": {"label": "the last complete month against the same month a year earlier", "kind": "year_over_year"},
}


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

def comparison_windows(grain: str, first_period: str, last_period: str) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return (current_start, current_end, base_start, base_end) pairs, latest first.

    Ends are inclusive calendar days.  Only pairs fully inside the analyzed
    complete periods are returned; the first is the comparison reported.
    """
    first, last = pd.Timestamp(first_period), pd.Timestamp(last_period)
    pairs = []
    if grain == "month":
        current = last
        while True:
            base = current - pd.DateOffset(years=1)
            if base < first:
                break
            pairs.append(
                (current, current + pd.offsets.MonthEnd(0), base, base + pd.offsets.MonthEnd(0))
            )
            current = current - pd.DateOffset(months=1)
        return pairs
    if grain == "week":
        length = 28
        end = last + pd.Timedelta(days=6)
    elif grain == "day":
        length = 7
        end = last
    else:
        raise ValueError(f"Unknown grain: {grain}")
    step = pd.Timedelta(days=length)
    while True:
        current_start = end - step + pd.Timedelta(days=1)
        base_start = current_start - step
        if base_start < first:
            break
        pairs.append((current_start, end, base_start, current_start - pd.Timedelta(days=1)))
        end = end - step
    return pairs


def _in_window(times: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (times >= start) & (times < end + pd.Timedelta(days=1))


def _window_total(values: pd.Series, mask: pd.Series, days: int, summed: bool, per_day: bool) -> float:
    selected = values[mask]
    if summed:
        return float(selected.sum()) / (days if per_day else 1)
    return float(selected.mean())


# --------------------------------------------------------------------------
# Decomposition primitives
# --------------------------------------------------------------------------

def decompose_sum(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, Any]:
    """Exact additive decomposition: total change = sum of segment changes."""
    segments = sorted(set(before) | set(after))
    rows = []
    for segment in segments:
        b, a = float(before.get(segment, 0.0)), float(after.get(segment, 0.0))
        rows.append(
            {
                "segment": segment,
                "before": b,
                "after": a,
                "contribution": a - b,
                "relative_change": (a - b) / b if b > 0 and a >= 0 else None,
            }
        )
    total_before = float(sum(before.values()))
    total_after = float(sum(after.values()))
    return {"before": total_before, "after": total_after, "change": total_after - total_before, "segments": rows}


def decompose_mean(
    before: Mapping[str, tuple[int, float]],
    after: Mapping[str, tuple[int, float]],
) -> dict[str, Any]:
    """Exact symmetric shift-share of a change in an average.

    ``before``/``after`` map segment -> (row count, segment mean).  For every
    segment, mix = change in row share x average level and rate = change in
    level x average share; mix and rate sum exactly to the change in the
    overall average.  A segment present in only one window takes its observed
    level on both sides, so all of its effect is mix.
    """
    n_before = sum(count for count, _ in before.values())
    n_after = sum(count for count, _ in after.values())
    if n_before <= 0 or n_after <= 0:
        raise ValueError("Both windows need rows.")
    segments = sorted(set(before) | set(after))
    overall_before = sum(count * mean for count, mean in before.values() if count) / n_before
    overall_after = sum(count * mean for count, mean in after.values() if count) / n_after
    # Mix is measured against the overall level: gaining share only lowers
    # the average when the segment sits below it.  Row shares sum to one in
    # both windows, so centring leaves the total mix effect unchanged.
    reference = (overall_before + overall_after) / 2.0
    rows, mix_total, rate_total = [], 0.0, 0.0
    for segment in segments:
        count_b, mean_b = before.get(segment, (0, math.nan))
        count_a, mean_a = after.get(segment, (0, math.nan))
        if count_b == 0:
            mean_b = mean_a
        if count_a == 0:
            mean_a = mean_b
        share_b, share_a = count_b / n_before, count_a / n_after
        mix = (share_a - share_b) * ((mean_a + mean_b) / 2.0 - reference)
        rate = (mean_a - mean_b) * (share_a + share_b) / 2.0
        mix_total += mix
        rate_total += rate
        rows.append(
            {
                "segment": segment,
                "share_before": share_b,
                "share_after": share_a,
                "before": float(mean_b),
                "after": float(mean_a),
                "mix": float(mix),
                "rate": float(rate),
                "contribution": float(mix + rate),
            }
        )
    return {
        "before": float(overall_before),
        "after": float(overall_after),
        "change": float(overall_after - overall_before),
        "mix_effect": float(mix_total),
        "rate_effect": float(rate_total),
        "segments": rows,
    }


def _attach_driver_scores(segments: list[dict[str, Any]], change: float, base_shares: Mapping[str, float]) -> Optional[dict[str, Any]]:
    """Score how disproportionately each segment moved; return the main driver.

    A segment that is 30% of the business but carries 80% of a drop is a
    driver (excess 0.5).  When every segment moves in proportion to its size,
    no segment is singled out: the change is broad-based.
    """
    driver, best = None, -math.inf
    for row in segments:
        share_of_change = row["contribution"] / change if change != 0 else None
        base_share = float(base_shares.get(row["segment"], 0.0))
        row["share_of_change"] = share_of_change
        row["base_share"] = base_share
        row["excess"] = share_of_change - base_share if share_of_change is not None else None
        if row["excess"] is not None and row["contribution"] * change > 0 and row["excess"] > best:
            driver, best = row, row["excess"]
    return driver


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class DriverAnalysisEngine:
    """Decompose the latest change of key measures across categorical dimensions."""

    def __init__(
        self,
        df: pd.DataFrame,
        temporal_analysis: Mapping[str, Any],
        *,
        alpha: float = DEFAULT_ALPHA,
        column_profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        analysis_context: Optional[Mapping[str, Any]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df
        self.temporal = temporal_analysis if isinstance(temporal_analysis, Mapping) else {}
        self.alpha = float(alpha)
        self.column_profiles = dict(column_profiles) if column_profiles is not None else profile_dataframe_columns(df)
        self.analysis_context = dict(analysis_context or {})

    def _status(self, status: str, summary: str, **extra: Any) -> dict[str, Any]:
        return {
            "contract": DRIVER_CONTRACT,
            "status": status,
            "summary": summary,
            "alpha": self.alpha,
            "measures": extra.pop("measures", []),
            "skipped": extra.pop("skipped", []),
            **extra,
        }

    def _dimensions(self, time_column: str) -> list[Any]:
        excluded = set(self.analysis_context.get("excluded_columns") or [])
        focus = set(self.analysis_context.get("focus_columns") or [])
        columns = [
            column
            for column in columns_for_role(self.df, "category", profiles=self.column_profiles, require_variation=True)
            if str(column) not in excluded
            and str(column) != time_column
            and int((self.column_profiles.get(column) or {}).get("unique_count", 0)) <= MAX_DIMENSION_CARDINALITY
        ]
        columns.sort(key=lambda column: (str(column) not in focus, int(self.column_profiles[column]["unique_count"])))
        return columns[:MAX_DIMENSIONS]

    def _measure_order(self, series: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        focus = set(self.analysis_context.get("focus_columns") or [])
        excluded = set(self.analysis_context.get("excluded_columns") or [])
        usable = [item for item in series if item.get("column") not in excluded]
        # Focus first, then summed business totals, then the engine's order.
        usable.sort(key=lambda item: (item.get("column") not in focus, item.get("unit") not in SUMMED_UNITS))
        return usable[:MAX_MEASURES]

    def analyze(self) -> dict[str, Any]:
        if self.temporal.get("status") != "ok" or not self.temporal.get("series"):
            return self._status("not_applicable", "Driver analysis needs a dated table with analyzable measures.")
        time_column = str(self.temporal.get("time_column"))
        column_lookup = {str(column): column for column in self.df.columns}
        if time_column not in column_lookup:
            return self._status("not_applicable", "The time column is not available.")
        times = pd.to_datetime(self.df[column_lookup[time_column]], errors="coerce")
        if getattr(times.dt, "tz", None) is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        dimensions = self._dimensions(time_column)

        results, skipped = [], []
        for item in self._measure_order(list(self.temporal["series"])):
            column = column_lookup.get(str(item.get("column")))
            if column is None:
                continue
            try:
                result = self._analyze_measure(item, column, times, dimensions)
            except ValueError as error:
                skipped.append({"column": str(item.get("column")), "reason": "invalid", "message": str(error)})
                continue
            if result.get("skip_reason"):
                skipped.append({"column": str(item.get("column")), "reason": result["skip_reason"], "message": result["message"]})
                continue
            results.append(result)

        if not results:
            return self._status(
                "insufficient_data",
                "No measure has two comparable complete windows with enough rows.",
                skipped=skipped,
                dimensions=[str(column) for column in dimensions],
            )

        tested = [result["unusual"] for result in results if result["unusual"]]
        if tested:
            adjusted = benjamini_hochberg([entry["p_value"] for entry in tested])
            for entry, p_adjusted in zip(tested, adjusted):
                entry["adjusted_p_value"] = float(p_adjusted)
                entry["significant"] = bool(p_adjusted <= self.alpha)
        for result in results:
            result["material"] = self._material(result)

        flagged = sum(1 for result in results if result["material"])
        return self._status(
            "ok",
            f"Decomposed the latest change of {len(results)} measure(s) across {len(dimensions)} dimension(s); "
            f"{flagged} changed more than usual.",
            measures=results,
            skipped=skipped,
            dimensions=[str(column) for column in dimensions],
            methodology_notes=[
                "Daily data compares the last 7 days with the 7 before, weekly data the last 4 complete weeks with "
                "the 4 before, and monthly data the last complete month with the same month a year earlier.",
                "For summed measures, each segment's contribution is its own change, so contributions add up "
                "exactly to the total change.",
                "For averaged measures, a symmetric shift-share splits the change into a mix effect (rows moved "
                "between segments) and a rate effect (segments moved); the two add up exactly.",
                "A change counts as larger than usual when it stands out against the same comparison repeated over "
                "the history (robust z-score, Student t reference, false discovery rate across measures).",
                "Contributions show where a change happened, not what caused it.",
            ],
        )

    def _material(self, result: Mapping[str, Any]) -> bool:
        unusual = result.get("unusual")
        if unusual:
            return unusual.get("significant") is True
        relative = result.get("relative_change")
        # Without enough history, only large changes are reported, and the
        # report marks them as exploratory.
        return relative is not None and abs(relative) >= 2 * MIN_MATERIAL_CHANGE

    def _analyze_measure(
        self, item: Mapping[str, Any], column: Any, times: pd.Series, dimensions: list[Any]
    ) -> dict[str, Any]:
        grain = str(item.get("grain"))
        if grain not in _WINDOWS:
            raise ValueError(f"Unsupported grain {grain}")
        pairs = comparison_windows(grain, str(item.get("first_period")), str(item.get("last_period")))
        if not pairs:
            return {"skip_reason": "too_short", "message": "Not enough history for two comparable windows."}
        summed = item.get("unit") in SUMMED_UNITS
        values = pd.to_numeric(self.df[column], errors="coerce").astype(float)
        current_start, current_end, base_start, base_end = pairs[0]
        current_mask = _in_window(times, current_start, current_end) & np.isfinite(values)
        base_mask = _in_window(times, base_start, base_end) & np.isfinite(values)
        if int(current_mask.sum()) < MIN_ROWS_PER_WINDOW or int(base_mask.sum()) < MIN_ROWS_PER_WINDOW:
            return {"skip_reason": "too_few_rows", "message": f"Each window needs at least {MIN_ROWS_PER_WINDOW} rows."}

        # Windows of unequal length (a leap-year February) are compared per day.
        current_days = (current_end - current_start).days + 1
        base_days = (base_end - base_start).days + 1
        per_day = summed and current_days != base_days

        before = _window_total(values, base_mask, base_days, summed, per_day)
        after = _window_total(values, current_mask, current_days, summed, per_day)
        change = after - before
        # Percentages need a positive starting level; signed measures such as
        # profit are compared in absolute terms.
        relative = change / before if before > 0 and after >= 0 else None

        history = []
        for pair in pairs[1:]:
            c_mask = _in_window(times, pair[0], pair[1]) & np.isfinite(values)
            b_mask = _in_window(times, pair[2], pair[3]) & np.isfinite(values)
            if int(c_mask.sum()) < MIN_ROWS_PER_WINDOW or int(b_mask.sum()) < MIN_ROWS_PER_WINDOW:
                continue
            c_days, b_days = (pair[1] - pair[0]).days + 1, (pair[3] - pair[2]).days + 1
            pair_per_day = summed and c_days != b_days
            c_total = _window_total(values, c_mask, c_days, summed, pair_per_day)
            b_total = _window_total(values, b_mask, b_days, summed, pair_per_day)
            if relative is None:
                history.append(c_total - b_total)
            elif b_total > 0:
                history.append((c_total - b_total) / b_total)
        unusual = None
        if len(history) >= MIN_HISTORY_COMPARISONS:
            latest = relative if relative is not None else change
            found = robust_unusualness(history, latest)
            if found is not None:
                score, p_value, centre = found
                unusual = {
                    "method": "Robust z-score against the same comparison over the history (Student t reference)",
                    "history_comparisons": len(history),
                    "typical_change": float(centre),
                    "robust_z": float(score),
                    "p_value": float(p_value),
                }

        # Dimensions are decomposed even for a flat total: offsetting moves
        # (one region collapsing while another grows) are worth knowing.
        # Shares of a near-zero change explode (a segment can be "700% of the
        # change"), so drivers are only named for a material change.
        sized = abs(relative) >= MIN_MATERIAL_CHANGE if relative is not None else (
            before != 0 and abs(change) >= MIN_MATERIAL_CHANGE * abs(before)
        )
        dimension_results = []
        for dimension in dimensions:
            decomposition = self._decompose(
                dimension, values, base_mask, current_mask, summed, base_days, current_days, per_day, sized
            )
            if decomposition is not None:
                dimension_results.append(decomposition)
        # The most telling dimension first: a mix story, then the most
        # disproportionate segment.
        dimension_results.sort(
            key=lambda entry: (
                not entry.get("mix_story"),
                -entry["driver_excess"],
                -max((abs(row["contribution"]) for row in entry.get("offsetting") or []), default=0.0),
            )
        )

        return {
            "column": str(column),
            "unit": item.get("unit"),
            "kind": "sum" if summed else "average",
            "grain": grain,
            "window": _WINDOWS[grain]["label"],
            "comparison": _WINDOWS[grain]["kind"],
            "per_day": bool(per_day),
            "current_window": [current_start.date().isoformat(), current_end.date().isoformat()],
            "base_window": [base_start.date().isoformat(), base_end.date().isoformat()],
            "before": before,
            "after": after,
            "change": change,
            "relative_change": relative,
            "unusual": unusual,
            "dimensions": dimension_results,
        }

    def _segments(self, dimension: Any, mask: pd.Series) -> pd.Series:
        labels = self.df.loc[mask, dimension]
        return labels.map(lambda value: MISSING_SEGMENT if pd.isna(value) else " ".join(str(value).split())[:80])

    def _decompose(
        self,
        dimension: Any,
        values: pd.Series,
        base_mask: pd.Series,
        current_mask: pd.Series,
        summed: bool,
        base_days: int,
        current_days: int,
        per_day: bool,
        sized: bool,
    ) -> Optional[dict[str, Any]]:
        base_labels, current_labels = self._segments(dimension, base_mask), self._segments(dimension, current_mask)
        base_values, current_values = values[base_mask], values[current_mask]
        weight = base_values.abs().groupby(base_labels).sum().add(
            current_values.abs().groupby(current_labels).sum(), fill_value=0.0
        ) if summed else base_labels.value_counts().add(current_labels.value_counts(), fill_value=0)
        keep = set(weight.sort_values(ascending=False).index[:MAX_SEGMENTS_PER_DIMENSION])
        base_labels = base_labels.where(base_labels.isin(keep), OTHER_SEGMENT)
        current_labels = current_labels.where(current_labels.isin(keep), OTHER_SEGMENT)
        if base_labels.nunique() < 2 and current_labels.nunique() < 2:
            return None

        if summed:
            before = (base_values.groupby(base_labels).sum() / (base_days if per_day else 1)).to_dict()
            after = (current_values.groupby(current_labels).sum() / (current_days if per_day else 1)).to_dict()
            decomposition = decompose_sum(before, after)
        else:
            def stats_by(labels: pd.Series, data: pd.Series) -> dict[str, tuple[int, float]]:
                grouped = data.groupby(labels)
                return {str(key): (int(count), float(mean)) for key, count, mean in zip(grouped.size().index, grouped.size(), grouped.mean())}
            decomposition = decompose_mean(stats_by(base_labels, base_values), stats_by(current_labels, current_values))

        change = decomposition["change"]
        segments = decomposition["segments"]
        if summed:
            scale = sum(abs(row["before"]) for row in segments) or 1.0
            base_shares = {row["segment"]: abs(row["before"]) / scale for row in segments}
        else:
            base_shares = {row["segment"]: (row["share_before"] + row["share_after"]) / 2.0 for row in segments}
        driver = _attach_driver_scores(segments, change, base_shares)
        if not sized:
            driver = None
            for row in segments:
                row["share_of_change"] = row["excess"] = None
        segments.sort(key=lambda row: -abs(row["contribution"]))
        excess = float(driver["excess"]) if driver is not None else 0.0
        displayed = segments[:DISPLAYED_SEGMENTS]
        if driver is not None and driver not in displayed:
            displayed = displayed[:-1] + [driver]
        result = {
            "dimension": str(dimension),
            "segment_count": len(segments),
            "driver_excess": excess,
            "driver": dict(driver) if driver is not None and excess >= MIN_DRIVER_EXCESS else None,
            "segments": displayed,
        }
        if summed:
            # Largest segment-level swings hidden inside a flat total.
            total_before = abs(decomposition["before"]) or 1.0
            result["offsetting"] = [
                dict(row)
                for row in sorted(segments, key=lambda row: -abs(row["contribution"]))
                if abs(row["contribution"]) >= MIN_MATERIAL_CHANGE * total_before
            ][:2]
        if not summed:
            mix_driver = max(segments, key=lambda row: abs(row["mix"]), default=None)
            result["mix_driver"] = dict(mix_driver) if mix_driver is not None else None
            result["mix_effect"] = decomposition["mix_effect"]
            result["rate_effect"] = decomposition["rate_effect"]
            result["mix_share"] = decomposition["mix_effect"] / change if sized and change != 0 else None
            result["mix_story"] = bool(
                sized and change != 0 and decomposition["mix_effect"] * change > 0
                and decomposition["mix_effect"] / change >= MIX_STORY_SHARE
            )
        return result
