"""Auditable statistical validation for automatically generated hypotheses."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.column_semantics import columns_for_role, profile_dataframe_columns
from src.analysis_context import is_relevant
from src.structural_dependencies import additive_pair_dependency, detect_additive_dependencies
from src.semantic_contract import ALLOWED_UNITS, infer_column_unit
from src.group_statistics import (
    compare_group_means,
    newcombe_difference,
    rate_difference_test,
    wilson_interval,
)
from src.statistical_primitives import (
    benjamini_hochberg as _benjamini_hochberg,
    pearson_summary,
    welch_independent_summary,
)


DEFAULT_ALPHA = 0.05
MIN_PAIRED_SAMPLE = 8
MIN_GROUP_SAMPLE = 8
MAX_NUMERIC_COLUMNS = 8
MAX_CATEGORICAL_COLUMNS = 5
MAX_GROUP_LEVELS = 12
MAX_REPORTED_PAIRS = 6
MAGNITUDE_RANK = {"large": 0, "medium": 1, "small": 2, "negligible": 3}
_POSITIVE_LABELS = frozenset({"yes", "y", "true", "t", "1", "1.0"})


@dataclass
class StatisticalTestResult:
    """A JSON-serializable statistical result with effect and uncertainty."""

    test_id: str
    family: str
    title: str
    method: str
    null_hypothesis: str
    columns: List[str]
    sample_size: int
    statistic: float
    raw_p_value: float
    adjusted_p_value: Optional[float]
    significant: Optional[bool]
    effect_size: Dict[str, Any]
    confidence_interval: Dict[str, Any]
    estimates: Dict[str, Any]
    limitations: List[str]
    interpretation: str = ""
    structural_dependency: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalValidationEngine:
    """Scan a bounded set of relationships and control false discoveries.

    Numeric pairs use Pearson correlation with a Fisher-z confidence interval.
    Numeric outcomes split by a binary categorical column use Welch's t-test,
    a confidence interval for the mean difference and Hedges' g.  Columns with
    three to twelve adequately sized levels use Welch's ANOVA with
    Holm-adjusted pairwise Welch comparisons.  A two-level outcome compared
    across segments is tested as a difference in rates.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float = DEFAULT_ALPHA,
        *,
        column_profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        analysis_context: Optional[Mapping[str, Any]] = None,
        structural_dependencies: Optional[List[Mapping[str, Any]]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.df = df
        self.analysis_context = dict(analysis_context or {})
        self.alpha = float(alpha)
        self.column_profiles = (
            dict(column_profiles)
            if column_profiles is not None
            else profile_dataframe_columns(df)
        )
        self.structural_dependencies = (
            structural_dependencies if structural_dependencies is not None
            else detect_additive_dependencies(df, self.column_profiles, self.analysis_context)
        )

    def analyze(self, max_results: int = 6) -> Dict[str, Any]:
        """Return corrected test results, ordered by adjusted p-value."""
        if max_results < 1:
            return self._empty_report("No results requested.")

        temporal_columns = [
            str(column) for column in self.df.columns
            if self.column_profiles.get(column, {}).get("role") == "temporal"
        ]
        if temporal_columns and not self.df.empty:
            # A date is a risk indicator, not proof of serial dependence (or
            # independence). Do not run IID tests and hide the caveat later.
            return {
                "status": "inference_withheld",
                "inference_status": "withheld",
                "reason": "temporal_design_unverified",
                "temporal_columns": temporal_columns,
                "alpha": self.alpha,
                "correction": None,
                "candidate_test_count": 0,
                "displayed_test_count": 0,
                "significant_test_count": None,
                "summary": (
                    "Inferential validation withheld: temporal fields are present, "
                    "but independence of observations and the time-series design "
                    "have not been established. Ordinary Pearson/Welch p-values "
                    "and confidence intervals are not reported. Descriptive "
                    "calculations remain available; significance is not assessed."
                ),
                "tests": [],
                "methodology_notes": [
                    "A temporal field does not itself prove autocorrelation; this is a conservative eligibility rule.",
                    "Neither a low lag-1 autocorrelation nor shuffling rows establishes independence.",
                    "Review the sampling unit, repeated entities, serial dependence and seasonality before choosing an inferential method.",
                    "FDR correction cannot repair invalid individual p-values.",
                ],
            }

        candidates = (
            self._correlation_candidates()
            + self._group_candidates()
            + self._rate_candidates()
        )
        if not candidates:
            return self._empty_report(
                "No eligible hypotheses: more numeric variation or adequately sized groups are required."
            )

        adjusted = _benjamini_hochberg([item.raw_p_value for item in candidates])
        for result, adjusted_p_value in zip(candidates, adjusted):
            result.adjusted_p_value = adjusted_p_value
            result.significant = adjusted_p_value <= self.alpha
            dependency = (
                additive_pair_dependency(result.columns, self.structural_dependencies)
                if result.family == "correlation" else None
            )
            if dependency:
                result.structural_dependency = dict(dependency)
                result.limitations.insert(0, (
                    f"Observed arithmetic relationship: {dependency['equation']}. "
                    "The total/component correlation is not an independent discovery."
                ))
            if not result.significant and "pairwise" in result.estimates:
                result.estimates["pairwise"] = []
            result.interpretation = self._interpret(result)

        # Significant results lead, larger practical effects first. Ordering
        # by tiny adjusted p-values alone would rank a negligible difference in
        # a large file above a large one that is merely "very" significant.
        # Magnitude classes are comparable across families; raw effect values
        # are not (r, g, omega squared, V), so they only break remaining ties.
        candidates.sort(
            key=lambda item: (
                not is_relevant({"columns": item.columns}, self.analysis_context),
                bool(item.structural_dependency),
                not item.significant,
                MAGNITUDE_RANK.get(item.effect_size.get("magnitude"), len(MAGNITUDE_RANK)),
                item.adjusted_p_value if item.adjusted_p_value is not None else 1.0,
                -abs(float(item.effect_size["value"])),
                item.test_id,
            )
        )
        significant_count = sum(bool(item.significant) for item in candidates)
        displayed = _diversified(candidates, max_results)
        return {
            "status": "ok",
            "alpha": self.alpha,
            "correction": "Benjamini-Hochberg false discovery rate (FDR)",
            "candidate_test_count": len(candidates),
            "displayed_test_count": len(displayed),
            "significant_test_count": significant_count,
            "summary": (
                f"Evaluated {len(candidates)} automatically generated hypotheses; "
                f"{significant_count} remained significant at FDR {self.alpha:.0%}."
            ),
            "tests": [item.to_dict() for item in displayed],
            "methodology_notes": [
                "The sampling design and independence of observations have not been verified automatically; these tests are conditional on their assumptions.",
                "P-values are adjusted across every eligible hypothesis, not only displayed results.",
                (
                    f"The automatic scan is bounded to the first {MAX_NUMERIC_COLUMNS} eligible numeric "
                    f"and {MAX_CATEGORICAL_COLUMNS} categorical columns; group comparisons use levels with at "
                    f"least {MIN_GROUP_SAMPLE} rows and at most {MAX_GROUP_LEVELS} such levels."
                ),
                "Three or more groups use Welch's ANOVA (bootstrap-t p-value up to 5,000 rows) with family-wise adjusted pairwise differences and simultaneous intervals, reported only when the overall test survives FDR; a two-level outcome across segments is a rate comparison (chi-square when every expected count is at least 5, otherwise Fisher's exact or a seeded Monte Carlo exact test).",
                "Identifiers and temporal fields are excluded; numeric category codes are tested as categories, not measures.",
                "Confidence intervals quantify uncertainty; effect sizes quantify practical magnitude.",
                "Statistical association alone does not establish causality or business importance.",
                "Detected exact integer total/component relationships remain in the FDR family, but are labelled and are not promoted as independent discoveries. The bounded arithmetic scan is not an exhaustive leakage or dependency check.",
            ],
        }

    def evidence_insights(
        self,
        report: Optional[Dict[str, Any]] = None,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Convert the leading statistical tests to the existing insight contract."""
        report = report if report is not None else self.analyze(max_results=max_results)
        if report.get("inference_status") == "withheld":
            return []
        insights = []
        relevant_tests = [test for test in report.get("tests", [])
                          if is_relevant(test, self.analysis_context)
                          and not test.get("structural_dependency")
                          and not additive_pair_dependency(test.get("columns") or [], self.structural_dependencies)]
        for test in relevant_tests[:max_results]:
            adjusted_p = float(test["adjusted_p_value"])
            effect = test["effect_size"]
            interval = test["confidence_interval"]
            evidence = [
                f"Method: {test['method']}",
                (
                    f"Raw p-value: {_format_probability(test['raw_p_value'])}; "
                    f"FDR-adjusted p-value: {_format_probability(adjusted_p)}"
                ),
                (
                    f"{effect['name']}: {effect['value']:.3f} "
                    f"({effect['magnitude']} magnitude)"
                ),
                (
                    f"{interval['level']:.0%} CI: "
                    f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"
                ),
            ]
            insights.append(
                {
                    "insight_id": f"statistical-{test['test_id']}",
                    "category": "statistical-validation",
                    "title": test["title"],
                    "statement": test["interpretation"],
                    "evidence": evidence,
                    "recommendation": (
                        "Validate assumptions and reproduce the result on independent data before acting."
                        if test["significant"]
                        else "Treat the result as inconclusive; collect more data or test a pre-specified hypothesis."
                    ),
                    "confidence": _sample_support(int(test["sample_size"])),
                    "severity": "warning" if test["significant"] else "info",
                    "sample_size": int(test["sample_size"]),
                    "metrics": {
                        "raw_p_value": float(test["raw_p_value"]),
                        "adjusted_p_value": adjusted_p,
                        "effect_size": float(effect["value"]),
                        "ci_lower": float(interval["lower"]),
                        "ci_upper": float(interval["upper"]),
                    },
                    "type": "evidence",
                }
            )
        return insights

    def _correlation_candidates(self) -> List[StatisticalTestResult]:
        columns = self._numeric_columns()[:MAX_NUMERIC_COLUMNS]
        results = []
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                paired = pd.DataFrame(
                    {
                        "left": pd.to_numeric(self.df[left], errors="coerce"),
                        "right": pd.to_numeric(self.df[right], errors="coerce"),
                    }
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if (
                    len(paired) < MIN_PAIRED_SAMPLE
                    or paired["left"].nunique() < 2
                    or paired["right"].nunique() < 2
                ):
                    continue

                summary = pearson_summary(paired["left"], paired["right"])
                if summary is None:
                    continue
                correlation = summary.correlation
                lower, upper = summary.confidence_interval
                results.append(
                    StatisticalTestResult(
                        test_id=f"pearson-{left}-{right}",
                        family="correlation",
                        title=f"Linear association: {left} vs {right}",
                        method="Pearson correlation",
                        null_hypothesis="The population linear correlation is zero.",
                        columns=[str(left), str(right)],
                        sample_size=int(len(paired)),
                        statistic=round(float(correlation), 6),
                        raw_p_value=summary.p_value,
                        adjusted_p_value=None,
                        significant=None,
                        effect_size={
                            "name": "Pearson r",
                            "value": round(float(correlation), 6),
                            "magnitude": _correlation_magnitude(correlation),
                        },
                        confidence_interval={
                            "level": 0.95,
                            "lower": round(lower, 6),
                            "upper": round(upper, 6),
                            "parameter": "Pearson r",
                        },
                        estimates={"pearson_r": round(float(correlation), 6)},
                        limitations=[
                            "Pearson correlation measures linear association and can be sensitive to outliers.",
                            "Association does not establish causality; confounding may explain the pattern.",
                        ],
                    )
                )
        return results

    def _group_candidates(self) -> List[StatisticalTestResult]:
        numeric_columns = self._numeric_columns()[:MAX_NUMERIC_COLUMNS]
        categorical_columns = self._categorical_columns()[:MAX_CATEGORICAL_COLUMNS]
        results = []
        for category in categorical_columns:
            counts = self.df[category].dropna().value_counts()
            levels = list(counts.index)
            if len(levels) == 2:
                left_level, right_level = levels
                for numeric in numeric_columns:
                    left = _finite_numeric(
                        self.df.loc[self.df[category] == left_level, numeric]
                    )
                    right = _finite_numeric(
                        self.df.loc[self.df[category] == right_level, numeric]
                    )
                    if len(left) < MIN_GROUP_SAMPLE or len(right) < MIN_GROUP_SAMPLE:
                        continue
                    result = self._welch_result(
                        numeric,
                        category,
                        left_level,
                        right_level,
                        left,
                        right,
                    )
                    if result is not None:
                        results.append(result)
                continue
            sized = [level for level in levels if int(counts[level]) >= MIN_GROUP_SAMPLE]
            if len(sized) < 3 or len(sized) > MAX_GROUP_LEVELS:
                continue
            excluded_rows = int(counts.sum() - counts[sized].sum())
            for numeric in numeric_columns:
                values = _finite_numeric(self.df[numeric])
                labels = self.df.loc[values.index, category]
                groups = [
                    (level, values[labels == level].to_numpy(dtype=float))
                    for level in sized
                ]
                groups = [item for item in groups if len(item[1]) >= MIN_GROUP_SAMPLE]
                if len(groups) < 3:
                    continue
                result = self._anova_result(numeric, category, groups, excluded_rows)
                if result is not None:
                    results.append(result)
        return results

    def _anova_result(
        self,
        numeric: Any,
        category: Any,
        groups: List[Any],
        excluded_rows: int,
    ) -> Optional[StatisticalTestResult]:
        names = [str(level) for level, _ in groups]
        arrays = [values for _, values in groups]
        comparison = compare_group_means(arrays)
        if comparison is None:
            return None
        summary, pairs = comparison.anova, comparison.pairs
        order = sorted(range(len(names)), key=lambda index: (-summary.means[index], names[index]))
        top, bottom = order[0], order[-1]
        extreme = next(
            pair for pair in pairs if {pair.left, pair.right} == {top, bottom}
        )
        sign = 1.0 if extreme.left == top else -1.0
        lower, upper = sorted(
            (sign * extreme.confidence_interval[0], sign * extreme.confidence_interval[1])
        )
        pairwise = []
        for pair in sorted(pairs, key=lambda item: (item.adjusted_p_value, -abs(item.difference))):
            if pair.adjusted_p_value > self.alpha:
                continue
            low_ci, high_ci = pair.confidence_interval
            if pair.difference > 0:
                high, low = pair.left, pair.right
            else:
                high, low = pair.right, pair.left
                low_ci, high_ci = -high_ci, -low_ci
            pairwise.append({
                "higher": names[high],
                "lower": names[low],
                "difference": round(abs(pair.difference), 6),
                "adjusted_p_value": pair.adjusted_p_value,
                "ci_lower": round(low_ci, 6),
                "ci_upper": round(high_ci, 6),
            })
        limitations = [
            "Welch's ANOVA compares group means without assuming equal variances; it assumes independent observations.",
            "Means are sensitive to outliers; a few very large values can drive a group difference.",
            "Pairwise differences are family-wise adjusted, and their intervals hold simultaneously across all pairs.",
            "Association with a segment does not establish that the segment causes the difference.",
        ]
        if excluded_rows:
            limitations.append(
                f"{excluded_rows} rows in levels with fewer than {MIN_GROUP_SAMPLE} observations were not compared."
            )
        return StatisticalTestResult(
            test_id=f"welch-anova-{numeric}-{category}",
            family="multi-group-comparison",
            title=f"Group differences: {numeric} across {category}",
            method=comparison.method,
            null_hypothesis=f"Mean {numeric} is equal in every {category} group.",
            columns=[str(numeric), str(category)],
            sample_size=int(sum(summary.sizes)),
            statistic=round(summary.statistic, 6),
            raw_p_value=comparison.p_value,
            adjusted_p_value=None,
            significant=None,
            effect_size={
                "name": "omega squared",
                "value": round(summary.omega_squared, 6),
                "magnitude": _variance_explained_magnitude(summary.omega_squared),
            },
            confidence_interval={
                "level": 0.95,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "parameter": f"Mean difference ({names[top]} - {names[bottom]}), simultaneous",
            },
            estimates={
                "group_column": str(category),
                "value_column": str(numeric),
                "unit": self._unit(numeric),
                "group_count": len(names),
                "groups": [
                    {"group": names[index], "size": summary.sizes[index], "mean": round(summary.means[index], 6)}
                    for index in order
                ],
                "highest_group": names[top],
                "lowest_group": names[bottom],
                "highest_mean": round(summary.means[top], 6),
                "lowest_mean": round(summary.means[bottom], 6),
                "mean_difference": round(summary.means[top] - summary.means[bottom], 6),
                "df_between": round(summary.df_between, 3),
                "df_within": round(summary.df_within, 3),
                "pairs_tested": len(pairs),
                "pairwise": pairwise[:MAX_REPORTED_PAIRS],
                "excluded_rows": excluded_rows,
            },
            limitations=limitations,
        )

    def _unit(self, column: Any) -> str:
        profile = self.column_profiles.get(column) or {}
        confirmed = profile.get("unit")
        if confirmed in ALLOWED_UNITS and confirmed != "none":
            return str(confirmed)
        inferred = infer_column_unit(self.df, column, profile)
        return inferred if inferred != "none" else "number"

    def _rate_candidates(self) -> List[StatisticalTestResult]:
        categorical_columns = self._categorical_columns()[:MAX_CATEGORICAL_COLUMNS]
        binary = {}
        for column in categorical_columns:
            counts = self.df[column].dropna().value_counts()
            if len(counts) == 2 and int(counts.min()) >= MIN_GROUP_SAMPLE:
                binary[column] = counts
        focus = set(self.analysis_context.get("focus_columns") or [])
        results = []
        seen = set()
        for outcome, outcome_counts in binary.items():
            for segment in categorical_columns:
                if segment == outcome:
                    continue
                pair = frozenset((str(outcome), str(segment)))
                if segment in binary:
                    if pair in seen:
                        continue
                    if _outcome_preference(segment, binary[segment], focus) < _outcome_preference(
                        outcome, outcome_counts, focus
                    ):
                        continue
                seen.add(pair)
                result = self._rate_result(outcome, outcome_counts, segment)
                if result is not None:
                    results.append(result)
        return results

    def _rate_result(self, outcome: Any, outcome_counts: pd.Series, segment: Any) -> Optional[StatisticalTestResult]:
        frame = self.df[[outcome, segment]].dropna()
        segment_counts = frame[segment].value_counts()
        sized = [level for level in segment_counts.index if int(segment_counts[level]) >= MIN_GROUP_SAMPLE]
        if len(sized) < 2 or len(sized) > MAX_GROUP_LEVELS:
            return None
        positive = _positive_level(outcome_counts)
        frame = frame[frame[segment].isin(sized)]
        events = (frame[outcome] == positive).groupby(frame[segment]).sum()
        sizes = frame[segment].value_counts()
        names = [str(level) for level in sized]
        successes = [int(events.get(level, 0)) for level in sized]
        totals = [int(sizes[level]) for level in sized]
        test = rate_difference_test(successes, totals)
        if test is None:
            return None
        rates = [x / n for x, n in zip(successes, totals)]
        order = sorted(range(len(names)), key=lambda index: (-rates[index], names[index]))
        top, bottom = order[0], order[-1]
        pair_count = len(names) * (len(names) - 1) // 2
        level = 1.0 - (1.0 - 0.95) / pair_count
        lower, upper = newcombe_difference(successes[top], totals[top], successes[bottom], totals[bottom], level)
        segments = []
        for index in order:
            low, high = wilson_interval(successes[index], totals[index])
            segments.append({
                "segment": names[index],
                "size": totals[index],
                "events": successes[index],
                "rate": round(rates[index], 6),
                "ci_lower": round(low, 6),
                "ci_upper": round(high, 6),
            })
        excluded_rows = int(segment_counts.sum() - sum(totals))
        limitations = [
            "Rows are assumed to be independent observations of the outcome.",
            "Rate differences between segments are associations; other differences between the segments may explain them.",
            f"The interval for the highest-minus-lowest rate is simultaneous across all {pair_count} segment pairs.",
        ]
        if excluded_rows:
            limitations.append(
                f"{excluded_rows} rows in segments with fewer than {MIN_GROUP_SAMPLE} observations were not compared."
            )
        return StatisticalTestResult(
            test_id=f"rate-{outcome}-{segment}",
            family="rate-comparison",
            title=f"Rate differences: {outcome} = {positive} across {segment}",
            method=test.method,
            null_hypothesis=f"The share of {outcome} = {positive} is equal in every {segment} group.",
            columns=[str(outcome), str(segment)],
            sample_size=int(sum(totals)),
            statistic=round(test.statistic, 6),
            raw_p_value=test.p_value,
            adjusted_p_value=None,
            significant=None,
            effect_size={
                "name": "Cramer's V (bias-corrected)",
                "value": round(test.cramers_v, 6),
                "magnitude": _correlation_magnitude(test.cramers_v),
            },
            confidence_interval={
                "level": 0.95,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "parameter": f"Rate difference ({names[top]} - {names[bottom]}), simultaneous",
            },
            estimates={
                "outcome_column": str(outcome),
                "positive_level": str(positive),
                "segment_column": str(segment),
                "overall_rate": round(sum(successes) / sum(totals), 6),
                "segments": segments,
                "highest_segment": names[top],
                "lowest_segment": names[bottom],
                "highest_rate": round(rates[top], 6),
                "lowest_rate": round(rates[bottom], 6),
                "rate_difference": round(rates[top] - rates[bottom], 6),
                "min_expected_count": round(test.min_expected_count, 3),
                "excluded_rows": excluded_rows,
            },
            limitations=limitations,
        )

    def _welch_result(
        self,
        numeric: Any,
        category: Any,
        left_level: Any,
        right_level: Any,
        left: pd.Series,
        right: pd.Series,
    ) -> Optional[StatisticalTestResult]:
        left_values = left.to_numpy(dtype=float)
        right_values = right.to_numpy(dtype=float)
        summary = welch_independent_summary(left_values, right_values)
        if summary is None:
            return None
        left_name = str(left_level)
        right_name = str(right_level)
        return StatisticalTestResult(
            test_id=f"welch-{numeric}-{category}-{left_name}-{right_name}",
            family="group-comparison",
            title=f"Group difference: {numeric} by {category}",
            method="Welch's independent-samples t-test",
            null_hypothesis=f"Mean {numeric} is equal for {left_name} and {right_name}.",
            columns=[str(numeric), str(category)],
            sample_size=int(len(left_values) + len(right_values)),
            statistic=round(summary.statistic, 6),
            raw_p_value=summary.p_value,
            adjusted_p_value=None,
            significant=None,
            effect_size={
                "name": "Hedges' g",
                "value": round(summary.hedges_g, 6),
                "magnitude": _standardized_effect_magnitude(summary.hedges_g),
            },
            confidence_interval={
                "level": 0.95,
                "lower": round(summary.confidence_interval[0], 6),
                "upper": round(summary.confidence_interval[1], 6),
                "parameter": f"Mean difference ({left_name} - {right_name})",
            },
            estimates={
                "group_column": str(category),
                "value_column": str(numeric),
                "group_a": left_name,
                "group_b": right_name,
                "group_a_size": int(len(left_values)),
                "group_b_size": int(len(right_values)),
                "group_a_mean": round(float(np.mean(left_values)), 6),
                "group_b_mean": round(float(np.mean(right_values)), 6),
                "mean_difference": round(summary.mean_difference, 6),
                "welch_degrees_of_freedom": round(summary.degrees_of_freedom, 3),
            },
            limitations=[
                "Welch's t-test assumes independent observations and estimates a difference in means.",
                "Outliers, dependence, selection bias or confounding can invalidate the conclusion.",
            ],
        )

    def _interpret(self, result: StatisticalTestResult) -> str:
        if result.structural_dependency:
            return (
                f"Observed arithmetic relationship: {result.structural_dependency['equation']}. "
                "This total/component correlation is not an independent discovery or evidence "
                "of an external driver. Test statistics are retained for audit, not promoted "
                "as a separate explanatory finding."
            )
        adjusted = _format_probability(result.adjusted_p_value or 0.0)
        effect = result.effect_size
        interval = result.confidence_interval
        if result.family == "correlation":
            if result.significant:
                return (
                    f"The linear association remains statistically detectable after FDR correction "
                    f"(adjusted p={adjusted}, {effect['name']}={effect['value']:.3f}, "
                    f"95% CI [{interval['lower']:.3f}, {interval['upper']:.3f}]). "
                    "This does not imply causation."
                )
            return (
                f"The scan did not find sufficient evidence of a linear association after FDR "
                f"correction (adjusted p={adjusted}). This does not prove that the variables are unrelated."
            )

        estimates = result.estimates
        if result.family == "multi-group-comparison":
            if result.significant:
                pairs = estimates.get("pairwise") or []
                detail = (
                    f" {len(pairs)} of {estimates['pairs_tested']} pairs differ after family-wise adjustment."
                    if pairs else " No single pair is distinguishable after family-wise adjustment."
                )
                return (
                    f"Mean {estimates['value_column']} differs across {estimates['group_column']} after FDR correction "
                    f"(adjusted p={adjusted}, {effect['name']}={effect['value']:.3f}). Highest: "
                    f"{estimates['highest_group']} ({estimates['highest_mean']:.4g}); lowest: "
                    f"{estimates['lowest_group']} ({estimates['lowest_mean']:.4g}); simultaneous 95% CI for the gap "
                    f"[{interval['lower']:.4g}, {interval['upper']:.4g}].{detail}"
                )
            return (
                f"The scan did not find sufficient evidence that mean {estimates['value_column']} differs across "
                f"{estimates['group_column']} after FDR correction (adjusted p={adjusted}). "
                "This does not prove that the groups are equivalent."
            )
        if result.family == "rate-comparison":
            label = f"{estimates['outcome_column']} = {estimates['positive_level']}"
            if result.significant:
                return (
                    f"The share of {label} differs across {estimates['segment_column']} after FDR correction "
                    f"(adjusted p={adjusted}, {effect['name']}={effect['value']:.3f}). Highest: "
                    f"{estimates['highest_segment']} ({estimates['highest_rate']:.1%}); lowest: "
                    f"{estimates['lowest_segment']} ({estimates['lowest_rate']:.1%}); simultaneous 95% CI for the gap "
                    f"[{interval['lower']:.1%}, {interval['upper']:.1%}]."
                )
            return (
                f"The scan did not find sufficient evidence that the share of {label} differs across "
                f"{estimates['segment_column']} after FDR correction (adjusted p={adjusted}). "
                "This does not prove that the rates are equal."
            )
        if result.significant:
            return (
                f"The mean difference remains statistically detectable after FDR correction "
                f"(adjusted p={adjusted}, {effect['name']}={effect['value']:.3f}; "
                f"95% CI for {estimates['group_a']} - {estimates['group_b']}: "
                f"[{interval['lower']:.3f}, {interval['upper']:.3f}])."
            )
        return (
            f"The scan did not find sufficient evidence of a mean difference after FDR correction "
            f"(adjusted p={adjusted}). This does not prove that the groups are equivalent."
        )

    def _numeric_columns(self) -> List[Any]:
        columns = columns_for_role(
            self.df,
            "measure",
            profiles=self.column_profiles,
            require_variation=True,
        )
        eligible = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(self.df[column])
            and not pd.api.types.is_bool_dtype(self.df[column])
            and not pd.api.types.is_complex_dtype(self.df[column])
            and self.df[column].replace([np.inf, -np.inf], np.nan).nunique(dropna=True) >= 2
        ]
        excluded = set(self.analysis_context.get("excluded_columns") or [])
        focus = set(self.analysis_context.get("focus_columns") or [])
        return sorted(
            [column for column in eligible if str(column) not in excluded],
            key=lambda column: str(column) not in focus,
        )

    def _categorical_columns(self) -> List[Any]:
        columns = columns_for_role(
            self.df,
            "category",
            profiles=self.column_profiles,
            require_variation=True,
        )
        excluded = set(self.analysis_context.get("excluded_columns") or [])
        requested = set(self.analysis_context.get("group_columns") or [])
        return sorted(
            [column for column in columns if str(column) not in excluded],
            key=lambda column: str(column) not in requested,
        )

    def _empty_report(self, message: str) -> Dict[str, Any]:
        return {
            "status": "insufficient_data",
            "alpha": self.alpha,
            "correction": "Benjamini-Hochberg false discovery rate (FDR)",
            "candidate_test_count": 0,
            "displayed_test_count": 0,
            "significant_test_count": 0,
            "summary": message,
            "tests": [],
            "methodology_notes": [
                f"Correlation requires at least {MIN_PAIRED_SAMPLE} paired observations.",
                f"Group comparison requires at least {MIN_GROUP_SAMPLE} observations per group.",
                f"Group comparison uses categorical columns with 2 to {MAX_GROUP_LEVELS} levels of at least {MIN_GROUP_SAMPLE} rows.",
            ],
        }


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _diversified(ranked: List[StatisticalTestResult], limit: int) -> List[StatisticalTestResult]:
    """Keep the ranking, but take one result per family in turn.

    Otherwise a file with many strongly correlated measures fills every slot
    with correlations and hides the segment comparisons a reader asks about.
    Significant results still come before inconclusive ones.
    """
    selected: List[StatisticalTestResult] = []
    for tier in (
        [item for item in ranked if item.significant],
        [item for item in ranked if not item.significant],
    ):
        queues: Dict[str, List[StatisticalTestResult]] = {}
        for item in tier:
            queues.setdefault(item.family, []).append(item)
        while queues and len(selected) < limit:
            for family in list(queues):
                if len(selected) >= limit:
                    break
                selected.append(queues[family].pop(0))
                if not queues[family]:
                    del queues[family]
    return selected


def _positive_level(counts: pd.Series) -> Any:
    """The level counted as the event: an explicit yes/true/1, else the rarer one."""
    levels = list(counts.index)
    explicit = [level for level in levels if str(level).strip().lower() in _POSITIVE_LABELS]
    if len(explicit) == 1:
        return explicit[0]
    return sorted(levels, key=lambda level: (int(counts[level]), str(level)))[0]


def _outcome_preference(column: Any, counts: pd.Series, focus: set) -> tuple:
    """Lower sorts first: the focus column, then an explicit yes/no, then the rarer event."""
    explicit = any(str(level).strip().lower() in _POSITIVE_LABELS for level in counts.index)
    return (str(column) not in focus, not explicit, int(counts.min()) / int(counts.sum()), str(column))


def _variance_explained_magnitude(value: float) -> str:
    magnitude = float(value)
    if magnitude >= 0.14:
        return "large"
    if magnitude >= 0.06:
        return "medium"
    if magnitude >= 0.01:
        return "small"
    return "negligible"


def _correlation_magnitude(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 0.50:
        return "large"
    if magnitude >= 0.30:
        return "medium"
    if magnitude >= 0.10:
        return "small"
    return "negligible"


def _standardized_effect_magnitude(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 0.80:
        return "large"
    if magnitude >= 0.50:
        return "medium"
    if magnitude >= 0.20:
        return "small"
    return "negligible"


def _format_probability(value: float) -> str:
    value = float(value)
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _sample_support(sample_size: int) -> str:
    if sample_size >= 100:
        return "high"
    if sample_size >= 30:
        return "medium"
    return "low"
