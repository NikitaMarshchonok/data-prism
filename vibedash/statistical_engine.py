"""Auditable statistical validation for automatically generated hypotheses."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.column_semantics import columns_for_role, profile_dataframe_columns
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalValidationEngine:
    """Scan a bounded set of relationships and control false discoveries.

    Numeric pairs use Pearson correlation with a Fisher-z confidence interval.
    Numeric outcomes split by a binary categorical column use Welch's t-test,
    a confidence interval for the mean difference and Hedges' g.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float = DEFAULT_ALPHA,
        *,
        column_profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.df = df
        self.alpha = float(alpha)
        self.column_profiles = (
            dict(column_profiles)
            if column_profiles is not None
            else profile_dataframe_columns(df)
        )

    def analyze(self, max_results: int = 6) -> Dict[str, Any]:
        """Return corrected test results, ordered by adjusted p-value."""
        if max_results < 1:
            return self._empty_report("No results requested.")

        candidates = self._correlation_candidates() + self._group_candidates()
        if not candidates:
            return self._empty_report(
                "No eligible hypotheses: more numeric variation or adequately sized binary groups are required."
            )

        adjusted = _benjamini_hochberg([item.raw_p_value for item in candidates])
        for result, adjusted_p_value in zip(candidates, adjusted):
            result.adjusted_p_value = adjusted_p_value
            result.significant = adjusted_p_value <= self.alpha
            result.interpretation = self._interpret(result)

        candidates.sort(
            key=lambda item: (
                item.adjusted_p_value if item.adjusted_p_value is not None else 1.0,
                -abs(float(item.effect_size["value"])),
                item.test_id,
            )
        )
        significant_count = sum(bool(item.significant) for item in candidates)
        displayed = candidates[:max_results]
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
                "P-values are adjusted across every eligible hypothesis, not only displayed results.",
                (
                    f"The automatic scan is bounded to the first {MAX_NUMERIC_COLUMNS} eligible numeric "
                    f"and {MAX_CATEGORICAL_COLUMNS} categorical columns; group tests require exactly two levels."
                ),
                "Identifiers and temporal fields are excluded; numeric category codes are tested as categories, not measures.",
                "Confidence intervals quantify uncertainty; effect sizes quantify practical magnitude.",
                "Statistical association alone does not establish causality or business importance.",
            ],
        }

    def evidence_insights(
        self,
        report: Optional[Dict[str, Any]] = None,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Convert the leading statistical tests to the existing insight contract."""
        report = report or self.analyze(max_results=max_results)
        insights = []
        for test in report.get("tests", [])[:max_results]:
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
            levels = list(self.df[category].dropna().value_counts().index)
            if len(levels) != 2:
                continue
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
        return results

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
        return [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(self.df[column])
            and not pd.api.types.is_bool_dtype(self.df[column])
            and not pd.api.types.is_complex_dtype(self.df[column])
            and self.df[column].replace([np.inf, -np.inf], np.nan).nunique(dropna=True) >= 2
        ]

    def _categorical_columns(self) -> List[Any]:
        return columns_for_role(
            self.df,
            "category",
            profiles=self.column_profiles,
            require_variation=True,
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
                "Automatic group comparison currently supports categorical columns with exactly two levels.",
            ],
        }


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


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
