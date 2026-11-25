"""Time-aware analysis for tables with a date column.

``StatisticalValidationEngine`` deliberately withholds IID tests when a table
has a temporal field, because neighbouring periods are not independent.  This
engine answers the questions a data scientist asks of a metric over time with
methods that remain valid under serial dependence:

1. The raw rows are aggregated to a regular period grid (day, week or month)
   using the column's unit: counts and currency are summed, ratios and other
   numbers are averaged.  Partial first and last periods are excluded, because
   a half-observed month looks like a collapse.
2. Trend: Theil–Sen slope with a Mann–Kendall test whose variance is inflated
   for autocorrelation (Hamed & Rao, 1998).  The slope confidence interval is
   derived from the same corrected variance.
3. Patterns (``temporal_patterns.py``): weekly or yearly rhythm, abrupt level
   shifts and an unusual latest period, each corrected for false discoveries
   across measures.

Everything is deterministic and bounded; no rows leave the process.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.column_semantics import columns_for_role, profile_dataframe_columns
from src.semantic_contract import ALLOWED_UNITS, infer_column_unit
from src.statistical_primitives import benjamini_hochberg

from .temporal_patterns import (
    MIN_LEVEL_SHIFT,
    MIN_SEASONAL_AMPLITUDE,
    last_period_check,
    level_shift_test,
    seasonality_test,
)


TEMPORAL_CONTRACT = "temporal-analysis-v1"
DEFAULT_ALPHA = 0.05
MAX_MEASURES = 5
MAX_ROWS = 100_000
MIN_PERIODS = 12
MAX_PERIODS = 1_000
# A grain is usable only when almost every interior period has data.  Sparse
# grids (for example business-day data at daily grain) move to a coarser grain
# instead of inventing zeros or silently skipping weekends.
MAX_EMPTY_PERIOD_SHARE = 0.10
MONTH_GRAIN_MIN_DAYS = 730
WEEK_GRAIN_MIN_DAYS = 112
SUMMED_UNITS = frozenset({"count", "currency"})
GRAINS = ("day", "week", "month")
PERIODS_PER_YEAR = {"day": 365.25, "week": 52.1775, "month": 12.0}
_PANDAS_FREQUENCY = {"day": "D", "week": "W-SUN", "month": "M"}


@dataclass
class PreparedSeries:
    """A regular, complete-period series ready for temporal tests."""

    column: str
    aggregation: str
    grain: str
    period_starts: list[pd.Timestamp]
    values: np.ndarray
    excluded_edge_periods: list[str] = field(default_factory=list)
    empty_periods: list[str] = field(default_factory=list)
    calendar_adjusted: bool = False

    @property
    def period_count(self) -> int:
        return int(np.isfinite(self.values).sum())


class TemporalSeriesError(ValueError):
    """A measure cannot be turned into an analyzable regular series."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# --------------------------------------------------------------------------
# Series preparation
# --------------------------------------------------------------------------

def _initial_grain(span_days: float) -> str:
    if span_days >= MONTH_GRAIN_MIN_DAYS:
        return "month"
    if span_days >= WEEK_GRAIN_MIN_DAYS:
        return "week"
    return "day"


def _period_bounds(period: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    return period.start_time.normalize(), period.end_time.normalize()


def prepare_series(
    timestamps: pd.Series,
    values: pd.Series,
    *,
    column: str,
    unit: str,
    grain: Optional[str] = None,
) -> PreparedSeries:
    """Aggregate raw rows to the finest regular grain the data supports.

    ``grain`` forces one grain (no fallback), for checks that need it.
    """
    if grain is not None and grain not in GRAINS:
        raise ValueError(f"Unknown grain: {grain}")
    times = pd.to_datetime(timestamps, errors="coerce")
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    numeric = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"time": times, "value": numeric})
    frame = frame[frame["time"].notna()]
    frame = frame[np.isfinite(frame["value"].astype(float))]
    if frame.empty:
        raise TemporalSeriesError("no_values", "No dated, finite values are available.")

    aggregation = "sum" if unit in SUMMED_UNITS else "mean"
    first_day, last_day = frame["time"].min().normalize(), frame["time"].max().normalize()
    span_days = float((last_day - first_day).days) + 1.0

    last_reason = ("too_short", "The series is too short for temporal analysis.")
    candidates = (grain,) if grain is not None else GRAINS[GRAINS.index(_initial_grain(span_days)):]
    for grain in candidates:
        periods = frame["time"].dt.to_period(_PANDAS_FREQUENCY[grain])
        grouped = frame["value"].groupby(periods)
        aggregated = grouped.sum() if aggregation == "sum" else grouped.mean()
        full_index = pd.period_range(periods.min(), periods.max(), freq=_PANDAS_FREQUENCY[grain])
        series = aggregated.reindex(full_index)

        excluded: list[str] = []
        # Edge periods are complete only when observations reach their
        # boundary day.  Interior periods are bounded by observed data on
        # both sides, so only the edges can be partially observed.
        if grain != "day" and len(series) >= 2:
            start, _ = _period_bounds(series.index[0])
            if first_day > start:
                excluded.append(str(series.index[0]))
                series = series.iloc[1:]
            _, end = _period_bounds(series.index[-1])
            if last_day < end:
                excluded.append(str(series.index[-1]))
                series = series.iloc[:-1]

        empty = [str(period) for period, value in series.items() if not np.isfinite(value)]
        interior = max(len(series), 1)
        if len(series) - len(empty) < MIN_PERIODS:
            last_reason = (
                "too_short",
                f"At most {len(series) - len(empty)} complete {grain}s are available; "
                f"{MIN_PERIODS} are required.",
            )
            continue
        if len(empty) / interior > MAX_EMPTY_PERIOD_SHARE:
            last_reason = (
                "sparse",
                f"{len(empty)} of {interior} {grain}s have no data.",
            )
            continue
        if len(series) > MAX_PERIODS:
            last_reason = ("too_long", f"More than {MAX_PERIODS} {grain}s; a coarser grain is used.")
            continue
        values = series.to_numpy(dtype=float)
        calendar_adjusted = False
        if aggregation == "sum" and grain == "month":
            # Months have 28-31 days, so raw monthly totals carry up to ~10%
            # of pure calendar variation.  Tests use the per-day rate.
            values = values / np.array([period.days_in_month for period in series.index], dtype=float)
            calendar_adjusted = True
        return PreparedSeries(
            column=column,
            aggregation=aggregation,
            grain=grain,
            period_starts=[period.start_time for period in series.index],
            values=values,
            excluded_edge_periods=excluded,
            empty_periods=empty,
            calendar_adjusted=calendar_adjusted,
        )
    raise TemporalSeriesError(*last_reason)


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------

def _mann_kendall_s(values: np.ndarray) -> tuple[float, float]:
    """Return Kendall's S and its tie-corrected variance under H0."""
    n = len(values)
    differences = np.sign(values[None, :] - values[:, None])
    s = float(np.triu(differences, k=1).sum())
    _, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1]
    variance = (n * (n - 1) * (2 * n + 5) - float(np.sum(ties * (ties - 1) * (2 * ties + 5)))) / 18.0
    return s, variance


def _hamed_rao_factor(values: np.ndarray, slope: float) -> float:
    """Variance inflation n/n* from significant autocorrelation of detrended ranks."""
    n = len(values)
    detrended = values - slope * np.arange(n)
    ranks = stats.rankdata(detrended)
    centred = ranks - ranks.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0 or n < 4:
        return 1.0
    bound = stats.norm.ppf(0.975) / math.sqrt(n)
    total = 0.0
    for lag in range(1, n - 2):
        r = float(np.dot(centred[:-lag], centred[lag:])) / denominator
        if abs(r) > bound:
            total += (n - lag) * (n - lag - 1) * (n - lag - 2) * r
    scale = 2.0 / (n * (n - 1) * (n - 2))
    significant_lags_factor = 1.0 + scale * total

    # Hamed-Rao keeps only lags that pass a significance screen, and the
    # sample autocorrelation is biased towards zero in short series, so on
    # its own the correction under-inflates the variance (measured: ~25%
    # false trends for AR(1) phi=0.6 at n=60).  A second estimate assumes an
    # AR(1) structure and extrapolates the lag-1 coefficient to every lag.
    # Around a known mean, E[r1] ~ rho - (1 + 3 rho) / n; residuals around an
    # *estimated* trend are biased further towards zero, so the first-order
    # correction is applied twice.  This choice is calibrated on the
    # synthetic quality gate (temporal_quality.py), not derived exactly.
    r1 = float(np.dot(centred[:-1], centred[1:])) / denominator
    rho = r1
    for _ in range(2):
        rho = (n * rho + 1.0) / (n - 3.0)
    rho = min(max(rho, 0.0), 0.95)
    lags = np.arange(1, n - 2)
    weights = (n - lags) * (n - lags - 1) * (n - lags - 2)
    ar1_factor = 1.0 + scale * float(np.sum(weights * rho**lags))

    # Negative autocorrelation would shrink the variance and make the test
    # anti-conservative; the correction is only ever allowed to widen it.
    return max(1.0, significant_lags_factor, ar1_factor)


def mann_kendall_trend(values: Sequence[float], *, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Autocorrelation-robust Mann–Kendall test with a Theil–Sen slope and CI."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < MIN_PERIODS:
        raise ValueError(f"At least {MIN_PERIODS} finite values are required.")
    x = np.arange(n, dtype=float)
    i, j = np.triu_indices(n, k=1)
    slopes = np.sort((y[j] - y[i]) / (x[j] - x[i]))
    slope = float(np.median(slopes))

    s, variance = _mann_kendall_s(y)
    factor = _hamed_rao_factor(y, slope)
    corrected = variance * factor
    if corrected <= 0:
        z, p_value = 0.0, 1.0
    else:
        z = (s - math.copysign(1.0, s)) / math.sqrt(corrected) if s != 0 else 0.0
        p_value = float(2.0 * stats.norm.sf(abs(z)))

    # Sen's confidence interval uses the same (corrected) variance so that the
    # interval and the test agree under serial dependence.
    c = stats.norm.ppf(1 - alpha / 2) * math.sqrt(max(corrected, 0.0))
    pairs = len(slopes)
    lower_index = int(math.floor((pairs - c) / 2.0))
    upper_index = int(math.ceil((pairs + c) / 2.0))
    lower = float(slopes[max(0, min(pairs - 1, lower_index))])
    upper = float(slopes[max(0, min(pairs - 1, upper_index))])
    intercept = float(np.median(y - slope * x))
    return {
        "method": "Mann-Kendall (Hamed-Rao autocorrelation correction) with Theil-Sen slope",
        "n": int(n),
        "s_statistic": s,
        "z_statistic": round(float(z), 6),
        "variance_inflation": round(float(factor), 6),
        "p_value": p_value,
        "slope_per_period": slope,
        "slope_ci": [lower, upper],
        "intercept": intercept,
        "median_level": float(np.median(y)),
    }


def _relative(value: float, level: float) -> Optional[float]:
    if not math.isfinite(value) or not math.isfinite(level) or level == 0:
        return None
    return value / abs(level)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class TemporalAnalysisEngine:
    """Analyze up to ``MAX_MEASURES`` measures over the table's time column."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        alpha: float = DEFAULT_ALPHA,
        column_profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        units: Optional[Mapping[str, str]] = None,
        analysis_context: Optional[Mapping[str, Any]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.df = df
        self.alpha = float(alpha)
        self.column_profiles = dict(column_profiles) if column_profiles is not None else profile_dataframe_columns(df)
        self.units = {str(key): str(value) for key, value in (units or {}).items()}
        self.analysis_context = dict(analysis_context or {})

    def _time_column(self) -> Optional[Any]:
        columns = columns_for_role(self.df, "temporal", profiles=self.column_profiles, require_variation=True)
        excluded = set(self.analysis_context.get("excluded_columns") or [])
        requested = set(self.analysis_context.get("time_columns") or [])
        columns = [column for column in columns if str(column) not in excluded]
        # A time axis confirmed in the semantic contract wins, then a column
        # the question mentions, then the profiler's order.
        columns.sort(
            key=lambda column: (
                (self.column_profiles.get(column) or {}).get("is_time") is not True,
                str(column) not in requested,
            )
        )
        return columns[0] if columns else None

    def _unit(self, column: Any) -> str:
        """Explicit units, then the confirmed contract, then the same inference the contract proposes."""
        explicit = self.units.get(str(column))
        if explicit in ALLOWED_UNITS and explicit != "none":
            return explicit
        profile = self.column_profiles.get(column) or {}
        confirmed = profile.get("unit")
        if confirmed in ALLOWED_UNITS and confirmed != "none":
            return confirmed
        inferred = infer_column_unit(self.df, column, profile)
        return inferred if inferred != "none" else "number"

    def _measures(self) -> list[Any]:
        columns = columns_for_role(self.df, "measure", profiles=self.column_profiles, require_variation=True)
        excluded = set(self.analysis_context.get("excluded_columns") or [])
        focus = set(self.analysis_context.get("focus_columns") or [])
        columns = [column for column in columns if str(column) not in excluded]
        columns.sort(key=lambda column: str(column) not in focus)
        return columns[:MAX_MEASURES]

    def _status(self, status: str, summary: str, **extra: Any) -> dict[str, Any]:
        return {
            "contract": TEMPORAL_CONTRACT,
            "status": status,
            "summary": summary,
            "alpha": self.alpha,
            "time_column": extra.pop("time_column", None),
            "series": extra.pop("series", []),
            "skipped": extra.pop("skipped", []),
            **extra,
        }

    def analyze(self) -> dict[str, Any]:
        if self.df.empty:
            return self._status("not_applicable", "The table is empty.")
        if len(self.df) > MAX_ROWS:
            return self._status("not_applicable", f"Temporal analysis supports at most {MAX_ROWS:,} rows.")
        time_column = self._time_column()
        if time_column is None:
            return self._status("not_applicable", "No usable date column was found.")
        measures = self._measures()
        if not measures:
            return self._status(
                "not_applicable", "No numeric measure varies over time.", time_column=str(time_column)
            )

        series_results: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for column in measures:
            unit = self._unit(column)
            try:
                prepared = prepare_series(
                    self.df[time_column], self.df[column], column=str(column), unit=unit
                )
                trend = mann_kendall_trend(prepared.values, alpha=self.alpha)
            except TemporalSeriesError as error:
                skipped.append({"column": str(column), "reason": error.reason, "message": str(error)})
                continue
            except ValueError as error:
                skipped.append({"column": str(column), "reason": "invalid", "message": str(error)})
                continue
            result = self._series_result(prepared, unit, trend)
            result.update(self._pattern_checks(time_column, column, unit, prepared))
            series_results.append(result)

        if not series_results:
            return self._status(
                "insufficient_data",
                "No measure has enough complete, regular periods for temporal analysis.",
                time_column=str(time_column),
                skipped=skipped,
            )

        adjusted = benjamini_hochberg([item["trend"]["p_value"] for item in series_results])
        for item, p_adjusted in zip(series_results, adjusted):
            trend = item["trend"]
            trend["adjusted_p_value"] = float(p_adjusted)
            lower, upper = trend["slope_ci"]
            significant = p_adjusted <= self.alpha and (lower > 0 or upper < 0)
            trend["significant"] = bool(significant)
            trend["direction"] = (
                ("increasing" if trend["slope_per_period"] > 0 else "decreasing") if significant else "no_trend"
            )

        self._adjust_family(series_results, "seasonality", MIN_SEASONAL_AMPLITUDE, "amplitude")
        self._adjust_family(series_results, "level_shift", MIN_LEVEL_SHIFT, "relative_shift")
        for item in series_results:
            shift = item.get("level_shift")
            if shift and shift.get("at_edge"):
                # Recent changes are covered by the latest-period and driver
                # checks; a shift whose date cannot be located is not reported.
                shift["significant"] = False
        self._adjust_family(series_results, "latest_period", None, None)

        detected = sum(1 for item in series_results if item["trend"]["significant"])
        return self._status(
            "ok",
            f"Analyzed {len(series_results)} measure(s) over {time_column}; "
            f"{detected} show a statistically reliable trend after correction.",
            time_column=str(time_column),
            series=series_results,
            skipped=skipped,
            correction="Benjamini-Hochberg false discovery rate (FDR)",
            methodology_notes=[
                "Rows are aggregated to complete calendar periods; partial first and last periods are excluded.",
                "Counts and currency are summed per period; ratios, percentages and other numbers are averaged.",
                "Monthly totals are tested as per-day rates so that 28- and 31-day months are comparable.",
                "Trend significance uses Mann-Kendall with the Hamed-Rao variance correction for autocorrelation.",
                "The slope is the Theil-Sen estimator; its interval uses the same corrected variance.",
                "A trend describes the analyzed period and is not a forecast.",
                "Weekly patterns are tested on daily values and yearly patterns on monthly values, after removing "
                "the trend, with a within-cycle permutation test; a pattern must also move the level by at least "
                f"{MIN_SEASONAL_AMPLITUDE:.0%} between its highest and lowest point.",
                "Level shifts use a CUSUM statistic with an AR(1) bootstrap null and must hold under both linear and "
                f"exponential trends and change the level by at least {MIN_LEVEL_SHIFT:.0%}.",
                "The latest complete period is compared with the average of the previous four, relative to how "
                "large such gaps normally are for the same measure.",
                "Each check is corrected for false discoveries across measures separately.",
            ],
        )

    def _pattern_checks(
        self, time_column: Any, column: Any, unit: str, prepared: PreparedSeries
    ) -> dict[str, Optional[dict[str, Any]]]:
        values, starts = prepared.values, prepared.period_starts
        seasonality = None
        if prepared.grain == "month":
            seasonality = seasonality_test(values, starts, cycle="month")
        else:
            # A weekly pattern lives in daily values even when the trend is
            # read at weekly grain, so the daily grid is built separately.
            daily = prepared
            if prepared.grain != "day":
                try:
                    daily = prepare_series(
                        self.df[time_column], self.df[column], column=str(column), unit=unit, grain="day"
                    )
                except TemporalSeriesError:
                    daily = None
            if daily is not None:
                seasonality = seasonality_test(daily.values, daily.period_starts, cycle="weekday")
        return {
            "seasonality": seasonality,
            "level_shift": level_shift_test(values, starts),
            "latest_period": last_period_check(values, starts),
        }

    def _adjust_family(
        self,
        series_results: list[dict[str, Any]],
        key: str,
        minimum: Optional[float],
        effect_key: Optional[str],
    ) -> None:
        tested = [item[key] for item in series_results if item.get(key)]
        if not tested:
            return
        adjusted = benjamini_hochberg([result["p_value"] for result in tested])
        for result, p_adjusted in zip(tested, adjusted):
            result["adjusted_p_value"] = float(p_adjusted)
            effect = result.get(effect_key) if effect_key else None
            large_enough = minimum is None or (effect is not None and abs(effect) >= minimum)
            result["significant"] = bool(p_adjusted <= self.alpha and large_enough)

    def _series_result(self, prepared: PreparedSeries, unit: str, trend: Mapping[str, Any]) -> dict[str, Any]:
        level = trend["median_level"]
        per_year = trend["slope_per_period"] * PERIODS_PER_YEAR[prepared.grain]
        lower, upper = trend["slope_ci"]
        span = trend["n"] - 1
        fitted_start = trend["intercept"]
        fitted_end = fitted_start + trend["slope_per_period"] * span
        # A percentage change is meaningful only for a positive level; signed
        # measures such as profit can cross zero, so they get no percentage.
        relative = fitted_start > 0 and fitted_end > 0
        change_over_period = {
            "periods": int(trend["n"]),
            "fitted_start": float(fitted_start),
            "fitted_end": float(fitted_end),
            "absolute_change": float(trend["slope_per_period"] * span),
            "relative_change": float(trend["slope_per_period"] * span / fitted_start) if relative else None,
            "relative_change_ci": (
                [float(lower * span / fitted_start), float(upper * span / fitted_start)] if relative else None
            ),
        }
        return {
            "column": prepared.column,
            "unit": unit,
            "aggregation": prepared.aggregation,
            "grain": prepared.grain,
            "calendar_adjusted": prepared.calendar_adjusted,
            "period_count": prepared.period_count,
            "first_period": prepared.period_starts[0].date().isoformat(),
            "last_period": prepared.period_starts[-1].date().isoformat(),
            "excluded_edge_periods": prepared.excluded_edge_periods,
            "empty_periods": prepared.empty_periods[:20],
            "empty_period_count": len(prepared.empty_periods),
            "change_over_period": change_over_period,
            "trend": {
                **dict(trend),
                "relative_slope_per_period": _relative(trend["slope_per_period"], level),
                "relative_change_per_year": _relative(per_year, level),
                "relative_slope_ci": [_relative(lower, level), _relative(upper, level)],
            },
        }
