"""Compare segments on dated data without assuming independent rows.

Rows that run over time are not independent, so the cross-sectional tests in
``statistical_engine`` withhold inference on dated tables.  This engine
answers the same business question ("is this segment really higher?") with a
design that tolerates time dependence:

1. Rows are averaged per (period, segment) cell.  Only periods in which every
   segment is observed are kept, so each period compares like with like.
2. Within a period, segment differences cancel everything the segments share:
   trend, seasonality, promotions, outages.
3. The remaining period-to-period dependence is handled by batch means: the
   periods are cut into ``q`` contiguous batches and the test treats the
   batch averages as the observations (Ibragimov and Mueller, 2010).  Several
   segments are tested jointly with Hotelling's T-squared on the batch means
   of the differences against a reference segment; pairs use t-tests on the
   batch means with Holm adjustment.

Numeric measures compare per-row averages; two-level outcomes (Yes/No)
compare event rates.  Results are corrected for false discoveries across all
measure-by-segment tests.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.column_semantics import columns_for_role, profile_dataframe_columns
from src.group_statistics import holm
from src.semantic_contract import ALLOWED_UNITS, infer_column_unit
from src.statistical_primitives import benjamini_hochberg


SEGMENT_TIME_CONTRACT = "segments-over-time-v1"
DEFAULT_ALPHA = 0.05
MAX_ROWS = 100_000
MAX_MEASURES = 4
MAX_OUTCOMES = 2
MAX_DIMENSIONS = 4
MAX_LEVELS = 8
MIN_ROWS_PER_LEVEL = 8
MIN_BATCHES = 8
MIN_PERIODS_PER_BATCH = 4
# Long histories get extra batches, each still at least this many periods
# long, so that degrees of freedom grow without making batches dependent.
LONG_BATCH_PERIODS = 8
MAX_BATCHES = 16
MIN_COMPLETE_SHARE = 0.5
MAX_REPORTED_PAIRS = 6
GRAINS = ("day", "week", "month")
_FREQUENCY = {"day": "D", "week": "W-SUN", "month": "M"}
_POSITIVE_LABELS = frozenset({"yes", "y", "true", "t", "1", "1.0"})


def batch_count(levels: int) -> int:
    """Batches needed for a joint test of ``levels`` segments (q > k - 1 + 2)."""
    return max(MIN_BATCHES, levels + 3)


def batch_means(values: np.ndarray, batches: int) -> np.ndarray:
    """Means of ``batches`` contiguous, near-equal blocks along axis 0."""
    return np.array([block.mean(axis=0) for block in np.array_split(values, batches)])


def joint_batch_test(cells: np.ndarray, batches: int) -> Optional[tuple[float, float, int, int]]:
    """Hotelling T-squared on batch means of differences to the first column.

    ``cells`` is periods x segments.  Returns (F, p, df1, df2) or ``None`` if
    the batch covariance is singular.
    """
    differences = cells[:, 1:] - cells[:, :1]
    means = batch_means(differences, batches)
    p = differences.shape[1]
    centre = means.mean(axis=0)
    covariance = np.atleast_2d(np.cov(means, rowvar=False, ddof=1))
    if not np.all(np.isfinite(covariance)) or np.linalg.matrix_rank(covariance) < p:
        return None
    t_squared = float(batches * centre @ np.linalg.solve(covariance, centre))
    df2 = batches - p
    statistic = (df2 / (p * (batches - 1))) * t_squared
    p_value = float(stats.f.sf(statistic, p, df2))
    if not np.isfinite(p_value):
        return None
    return float(statistic), p_value, p, df2


def pair_batch_test(difference: np.ndarray, batches: int, confidence_level: float) -> tuple[float, float, tuple[float, float]]:
    """t-test on batch means of one per-period difference: (mean, p, interval)."""
    means = batch_means(difference, batches)
    centre = float(difference.mean())
    spread = float(np.std(means, ddof=1))
    if spread <= 0:
        return centre, (0.0 if centre != 0 else 1.0), (centre, centre)
    se = spread / np.sqrt(batches)
    statistic = float(means.mean()) / se
    p_value = float(2.0 * stats.t.sf(abs(statistic), batches - 1))
    margin = float(stats.t.ppf(0.5 + confidence_level / 2.0, batches - 1)) * se
    return centre, p_value, (centre - margin, centre + margin)


def _relative_magnitude(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 0.25:
        return "large"
    if magnitude >= 0.10:
        return "medium"
    if magnitude >= 0.02:
        return "small"
    return "negligible"


def _positive_level(counts: pd.Series) -> Any:
    levels = list(counts.index)
    explicit = [level for level in levels if str(level).strip().lower() in _POSITIVE_LABELS]
    if len(explicit) == 1:
        return explicit[0]
    return sorted(levels, key=lambda level: (int(counts[level]), str(level)))[0]


class SegmentTimeEngine:
    """Time-aware segment comparisons for dated tables."""

    def __init__(
        self,
        df: pd.DataFrame,
        temporal_analysis: Optional[Mapping[str, Any]] = None,
        *,
        alpha: float = DEFAULT_ALPHA,
        column_profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        analysis_context: Optional[Mapping[str, Any]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.df = df
        self.temporal = dict(temporal_analysis or {})
        self.alpha = float(alpha)
        self.column_profiles = dict(column_profiles) if column_profiles is not None else profile_dataframe_columns(df)
        self.analysis_context = dict(analysis_context or {})

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _column(self, name: Any) -> Optional[Any]:
        for column in self.df.columns:
            if str(column) == str(name):
                return column
        return None

    def _excluded(self) -> set:
        return set(self.analysis_context.get("excluded_columns") or [])

    def _unit(self, column: Any) -> str:
        profile = self.column_profiles.get(column) or {}
        confirmed = profile.get("unit")
        if confirmed in ALLOWED_UNITS and confirmed != "none":
            return str(confirmed)
        inferred = infer_column_unit(self.df, column, profile)
        return inferred if inferred != "none" else "number"

    def _measures(self) -> list[Any]:
        focus = set(self.analysis_context.get("focus_columns") or [])
        columns = [
            column for column in columns_for_role(self.df, "measure", profiles=self.column_profiles, require_variation=True)
            if str(column) not in self._excluded()
            and pd.api.types.is_numeric_dtype(self.df[column])
            and not pd.api.types.is_bool_dtype(self.df[column])
            and not pd.api.types.is_complex_dtype(self.df[column])
        ]
        columns.sort(key=lambda column: str(column) not in focus)
        return columns[:MAX_MEASURES]

    def _categories(self) -> list[tuple[Any, pd.Series]]:
        result = []
        for column in columns_for_role(self.df, "category", profiles=self.column_profiles, require_variation=True):
            if str(column) in self._excluded():
                continue
            counts = self.df[column].dropna().value_counts()
            if 2 <= len(counts) <= MAX_LEVELS and int(counts.min()) >= MIN_ROWS_PER_LEVEL:
                result.append((column, counts))
        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _status(self, status: str, summary: str, **extra: Any) -> dict[str, Any]:
        return {
            "contract": SEGMENT_TIME_CONTRACT,
            "status": status,
            "summary": summary,
            "alpha": self.alpha,
            "time_column": extra.pop("time_column", None),
            "tests": extra.pop("tests", []),
            "skipped": extra.pop("skipped", []),
            **extra,
        }

    def analyze(self) -> dict[str, Any]:
        time_name = self.temporal.get("time_column")
        time_column = self._column(time_name) if time_name is not None else None
        if self.df.empty or time_column is None:
            return self._status("not_applicable", "No usable date column was found.")
        if len(self.df) > MAX_ROWS:
            return self._status("not_applicable", f"Segment comparison over time supports at most {MAX_ROWS:,} rows.")
        times = pd.to_datetime(self.df[time_column], errors="coerce")
        if getattr(times.dt, "tz", None) is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        if times.notna().sum() == 0:
            return self._status("not_applicable", "No usable date column was found.", time_column=str(time_column))

        categories = self._categories()
        focus = set(self.analysis_context.get("focus_columns") or [])
        groups = set(self.analysis_context.get("group_columns") or [])
        # A two-level column is an outcome when it reads as Yes/No (or is the
        # focus); otherwise, like Online/Store, it is a way to split segments.
        outcomes = [
            (column, counts) for column, counts in categories
            if len(counts) == 2 and (
                str(column) in focus
                or any(str(level).strip().lower() in _POSITIVE_LABELS for level in counts.index)
            )
        ]
        outcomes.sort(key=lambda item: (str(item[0]) not in focus, int(item[1].min()) / int(item[1].sum()), str(item[0])))
        outcomes = outcomes[:MAX_OUTCOMES]
        outcome_columns = {column for column, _ in outcomes}
        dimensions = [column for column, _ in categories if column not in outcome_columns]
        dimensions.sort(key=lambda column: (str(column) not in groups, str(column)))
        dimensions = dimensions[:MAX_DIMENSIONS]

        targets: list[dict[str, Any]] = []
        for column in self._measures():
            values = pd.to_numeric(self.df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            targets.append({"column": column, "kind": "average", "unit": self._unit(column), "values": values})
        for column, counts in outcomes:
            positive = _positive_level(counts)
            values = (self.df[column] == positive).astype(float).where(self.df[column].notna())
            targets.append({"column": column, "kind": "rate", "unit": "ratio", "values": values, "positive_level": positive})
        if not targets or not dimensions:
            return self._status(
                "not_applicable",
                "Comparing segments over time needs a measure or Yes/No outcome and a column with 2 to "
                f"{MAX_LEVELS} segments.",
                time_column=str(time_column),
            )

        tests: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for target in targets:
            for dimension in dimensions:
                if dimension == target["column"]:
                    continue
                result, reason = self._compare(times, target, dimension)
                if result is None:
                    skipped.append({"measure": str(target["column"]), "segment_column": str(dimension), "reason": reason})
                else:
                    tests.append(result)

        if not tests:
            return self._status(
                "insufficient_data",
                "Segments are not observed side by side in enough periods to compare them over time.",
                time_column=str(time_column),
                skipped=skipped,
                methodology_notes=self._notes(),
            )

        adjusted = benjamini_hochberg([test["p_value"] for test in tests])
        for test, p_adjusted in zip(tests, adjusted):
            test["adjusted_p_value"] = float(p_adjusted)
            test["significant"] = bool(p_adjusted <= self.alpha)
            if not test["significant"]:
                test["pairwise"] = []
        rank = {"large": 0, "medium": 1, "small": 2, "negligible": 3}
        tests.sort(key=lambda test: (
            not test["significant"],
            str(test["measure"]) not in focus,
            rank.get(test["magnitude"], 4),
            test["adjusted_p_value"],
            test["test_id"],
        ))
        significant = sum(test["significant"] for test in tests)
        return self._status(
            "ok",
            f"Compared segments over time in {len(tests)} measure-by-segment pairs; {significant} differ reliably "
            "after correction.",
            time_column=str(time_column),
            tests=tests,
            skipped=skipped,
            correction="Benjamini-Hochberg false discovery rate (FDR)",
            significant_test_count=significant,
            methodology_notes=self._notes(),
        )

    def _notes(self) -> list[str]:
        return [
            "Rows are averaged per period and segment; only periods in which every segment appears are compared.",
            "Differences between segments within the same period cancel shared trends, seasonality and one-off events.",
            f"Dependence between periods is handled with batch means: periods are cut into {MIN_BATCHES} to "
            f"{MAX_BATCHES} contiguous batches of at least {MIN_PERIODS_PER_BATCH} periods, and the batch averages are tested "
            "(Hotelling's T-squared for all segments, t-tests with Holm adjustment for pairs).",
            "Averages are per row: a segment with fewer, larger rows can have a higher average and a lower total.",
            "A reliable difference shows where results differ, not why; segments can differ in customers, prices or mix.",
        ]

    def _compare(self, times: pd.Series, target: Mapping[str, Any], dimension: Any) -> tuple[Optional[dict[str, Any]], str]:
        frame = pd.DataFrame({"time": times, "segment": self.df[dimension], "value": target["values"]}).dropna()
        counts = frame["segment"].value_counts()
        levels = sorted(
            [level for level in counts.index if int(counts[level]) >= MIN_ROWS_PER_LEVEL],
            key=lambda level: str(level),
        )
        if len(levels) < 2:
            return None, "too_few_segments"
        frame = frame[frame["segment"].isin(levels)]
        if frame["value"].nunique() < 2:
            return None, "no_variation"
        batches = batch_count(len(levels))
        minimum_periods = batches * MIN_PERIODS_PER_BATCH
        for grain in GRAINS:
            periods = frame["time"].dt.to_period(_FREQUENCY[grain])
            cells = frame.groupby([periods, frame["segment"]])["value"].mean().unstack()
            cells = cells.reindex(columns=levels)
            span = len(pd.period_range(periods.min(), periods.max(), freq=_FREQUENCY[grain]))
            complete = cells.dropna()
            if len(complete) >= minimum_periods and len(complete) / max(span, 1) >= MIN_COMPLETE_SHARE:
                used = max(batches, min(len(complete) // LONG_BATCH_PERIODS, MAX_BATCHES))
                result = self._result(target, dimension, levels, grain, complete, span, used)
                return (result, "") if result is not None else (None, "degenerate")
        return None, "too_few_shared_periods"

    def _result(
        self,
        target: Mapping[str, Any],
        dimension: Any,
        levels: list[Any],
        grain: str,
        complete: pd.DataFrame,
        span: int,
        batches: int,
    ) -> Optional[dict[str, Any]]:
        matrix = complete.to_numpy(dtype=float)
        joint = joint_batch_test(matrix, batches)
        if joint is None:
            return None
        statistic, p_value, df1, df2 = joint
        names = [str(level) for level in levels]
        averages = matrix.mean(axis=0)
        order = sorted(range(len(names)), key=lambda index: (-averages[index], names[index]))
        top, bottom = order[0], order[-1]
        pairs = list(combinations(range(len(names)), 2))
        level = 1.0 - (1.0 - 0.95) / len(pairs)
        raw = []
        for a, b in pairs:
            difference, pair_p, interval = pair_batch_test(matrix[:, a] - matrix[:, b], batches, level)
            raw.append((a, b, difference, pair_p, interval))
        adjusted = holm([item[3] for item in raw])
        pairwise = []
        gap_interval = (0.0, 0.0)
        for (a, b, difference, _, (low, high)), pair_adjusted in zip(raw, adjusted):
            if difference < 0:
                a, b, difference, low, high = b, a, -difference, -high, -low
            if {a, b} == {top, bottom}:
                gap_interval = (low, high)
            if pair_adjusted <= self.alpha:
                pairwise.append({
                    "higher": names[a], "lower": names[b], "difference": round(difference, 6),
                    "adjusted_p_value": pair_adjusted, "ci_lower": round(low, 6), "ci_upper": round(high, 6),
                })
        pairwise.sort(key=lambda pair: (pair["adjusted_p_value"], -pair["difference"]))
        overall = float(averages.mean())
        gap = float(averages[top] - averages[bottom])
        relative = gap / abs(overall) if overall != 0 else float("inf")
        consistency = float(np.mean(matrix[:, top] > matrix[:, bottom]))
        rows = int(complete.shape[0])
        measure = str(target["column"])
        result = {
            "test_id": f"segment-time-{measure}-{dimension}",
            "measure": measure,
            "kind": target["kind"],
            "unit": target["unit"],
            "segment_column": str(dimension),
            "grain": grain,
            "periods_compared": rows,
            "periods_in_range": int(span),
            "batches": batches,
            "method": "Batch-means Hotelling T-squared on within-period segment differences"
            if len(names) > 2 else "Batch-means t-test on within-period segment differences",
            "statistic": round(statistic, 6),
            "df": [df1, df2],
            "p_value": p_value,
            "segments": [{"segment": names[index], "average": round(float(averages[index]), 6)} for index in order],
            "highest_segment": names[top],
            "lowest_segment": names[bottom],
            "highest_average": round(float(averages[top]), 6),
            "lowest_average": round(float(averages[bottom]), 6),
            "gap": round(gap, 6),
            "gap_ci": [round(gap_interval[0], 6), round(gap_interval[1], 6)],
            "relative_gap": round(relative, 6) if np.isfinite(relative) else None,
            "magnitude": _relative_magnitude(relative) if np.isfinite(relative) else "large",
            "consistency": round(consistency, 4),
            "pairs_tested": len(pairs),
            "pairwise": pairwise[:MAX_REPORTED_PAIRS],
        }
        if target["kind"] == "rate":
            result["positive_level"] = str(target["positive_level"])
        return result
