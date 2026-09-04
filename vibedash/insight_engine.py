"""Deterministic, evidence-based insights for VibeDash.

The engine deliberately does not use an LLM.  It calculates facts first so an
LLM can later explain them without becoming the source of truth.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvidenceInsight:
    """A JSON-serializable analytical finding with its supporting evidence."""

    insight_id: str
    category: str
    title: str
    statement: str
    evidence: List[str]
    recommendation: str
    confidence: str
    severity: str
    sample_size: int
    metrics: Dict[str, Any]
    type: str = "evidence"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceBasedInsightEngine:
    """Produce conservative findings that can be traced back to the dataset."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df

    def generate(self, max_insights: int = 6) -> List[Dict[str, Any]]:
        if max_insights < 1:
            return []
        if self.df.empty:
            return [self._empty_dataset().to_dict()]

        findings: List[EvidenceInsight] = [self._dataset_overview(), self._data_quality()]
        for builder in (
            self._small_sample_warning,
            self._strongest_correlation,
            self._strongest_trend,
            self._largest_outlier_group,
            self._largest_category_concentration,
        ):
            finding = builder()
            if finding is not None:
                findings.append(finding)

        return [finding.to_dict() for finding in findings[:max_insights]]

    def _empty_dataset(self) -> EvidenceInsight:
        return EvidenceInsight(
            insight_id="dataset-empty",
            category="coverage",
            title="Dataset is empty",
            statement="No statistical conclusions can be calculated from this dataset.",
            evidence=["Observed rows: 0"],
            recommendation="Load at least one valid record before running analysis.",
            confidence="high",
            severity="critical",
            sample_size=0,
            metrics={"row_count": 0, "column_count": int(len(self.df.columns))},
        )

    def _dataset_overview(self) -> EvidenceInsight:
        numeric_count = len(self._numeric_columns())
        categorical_count = len(self._categorical_columns())
        row_count, column_count = self.df.shape
        return EvidenceInsight(
            insight_id="dataset-overview",
            category="coverage",
            title="Dataset coverage",
            statement=f"The analysis covers {row_count:,} rows and {column_count:,} columns.",
            evidence=[
                f"Numeric columns: {numeric_count}",
                f"Categorical columns: {categorical_count}",
            ],
            recommendation="Use these counts to confirm that the uploaded scope matches the intended analysis.",
            confidence="high",
            severity="info",
            sample_size=int(row_count),
            metrics={
                "row_count": int(row_count),
                "column_count": int(column_count),
                "numeric_column_count": int(numeric_count),
                "categorical_column_count": int(categorical_count),
            },
        )

    def _data_quality(self) -> EvidenceInsight:
        row_count, column_count = self.df.shape
        cell_count = row_count * column_count
        missing_cells = int(self.df.isna().sum().sum())
        missing_rate = missing_cells / cell_count if cell_count else 0.0
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_rate = duplicate_rows / row_count if row_count else 0.0
        constant_columns = [
            str(column)
            for column in self.df.columns
            if self.df[column].nunique(dropna=False) <= 1
        ]

        issues = []
        if missing_cells:
            issues.append(f"{missing_rate:.1%} of cells are missing")
        if duplicate_rows:
            issues.append(f"{duplicate_rate:.1%} of rows are duplicates")
        if constant_columns:
            issues.append(f"{len(constant_columns)} columns are constant")

        if missing_rate >= 0.20 or duplicate_rate >= 0.10:
            severity = "critical"
        elif issues:
            severity = "warning"
        else:
            severity = "info"

        statement = (
            "Data quality checks found: " + "; ".join(issues) + "."
            if issues
            else "No missing cells, duplicate rows, or constant columns were detected."
        )
        recommendation = (
            "Investigate the affected rows and columns before modelling; document any imputation or deduplication rule."
            if issues
            else "Continue monitoring these checks when the dataset is refreshed."
        )
        evidence = [
            f"Missing cells: {missing_cells:,} of {cell_count:,} ({missing_rate:.2%})",
            f"Duplicate rows: {duplicate_rows:,} of {row_count:,} ({duplicate_rate:.2%})",
        ]
        if constant_columns:
            evidence.append("Constant columns: " + ", ".join(constant_columns[:5]))

        return EvidenceInsight(
            insight_id="data-quality",
            category="quality",
            title="Data quality assessment",
            statement=statement,
            evidence=evidence,
            recommendation=recommendation,
            confidence="high",
            severity=severity,
            sample_size=int(row_count),
            metrics={
                "missing_cells": missing_cells,
                "missing_rate": round(float(missing_rate), 6),
                "duplicate_rows": duplicate_rows,
                "duplicate_rate": round(float(duplicate_rate), 6),
                "constant_column_count": len(constant_columns),
            },
        )

    def _strongest_correlation(self) -> Optional[EvidenceInsight]:
        columns = self._numeric_columns(exclude_identifiers=True)[:30]
        strongest = None
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                paired = self.df[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(paired) < 3 or paired[left].nunique() < 2 or paired[right].nunique() < 2:
                    continue
                correlation = float(paired[left].corr(paired[right]))
                if np.isnan(correlation):
                    continue
                candidate = (abs(correlation), correlation, str(left), str(right), len(paired))
                if strongest is None or candidate[0] > strongest[0]:
                    strongest = candidate

        if strongest is None or strongest[0] < 0.70:
            return None

        magnitude, correlation, left, right, sample_size = strongest
        direction = "positive" if correlation > 0 else "negative"
        confidence = self._sample_confidence(sample_size)
        severity = "warning" if magnitude >= 0.95 else "info"
        return EvidenceInsight(
            insight_id="strongest-correlation",
            category="relationship",
            title="Strong linear association",
            statement=(
                f"{left} and {right} have a strong {direction} linear association "
                f"(Pearson r={correlation:.3f})."
            ),
            evidence=[
                f"Paired non-missing observations: {sample_size:,}",
                f"Absolute correlation: {magnitude:.3f}",
            ],
            recommendation=(
                "Check domain logic and possible confounders. Treat correlation as association, not causation; "
                "for modelling, assess leakage and multicollinearity."
            ),
            confidence=confidence,
            severity=severity,
            sample_size=int(sample_size),
            metrics={
                "column_x": left,
                "column_y": right,
                "pearson_r": round(correlation, 6),
                "absolute_r": round(magnitude, 6),
            },
        )

    def _strongest_trend(self) -> Optional[EvidenceInsight]:
        time_column = self._time_column()
        if time_column is None:
            return None

        parsed_time = pd.to_datetime(self.df[time_column], errors="coerce")
        strongest = None
        for column in self._numeric_columns(exclude_identifiers=True)[:20]:
            series = pd.to_numeric(self.df[column], errors="coerce")
            frame = pd.DataFrame({"time": parsed_time, "value": series}).dropna().sort_values("time")
            if len(frame) < 8:
                continue
            grouped = frame.groupby("time", as_index=False)["value"].mean()
            if len(grouped) < 6:
                continue
            window = max(1, len(grouped) // 5)
            start_value = float(grouped["value"].head(window).mean())
            end_value = float(grouped["value"].tail(window).mean())
            scale = abs(start_value)
            if scale <= 1e-12:
                scale = float(grouped["value"].std())
            if not np.isfinite(scale) or scale <= 1e-12:
                continue
            relative_change = (end_value - start_value) / scale
            candidate = (
                abs(relative_change),
                relative_change,
                str(column),
                len(grouped),
                start_value,
                end_value,
            )
            if strongest is None or candidate[0] > strongest[0]:
                strongest = candidate

        if strongest is None or strongest[0] < 0.10:
            return None

        _, relative_change, column, periods, start_value, end_value = strongest
        direction = "increased" if relative_change > 0 else "decreased"
        return EvidenceInsight(
            insight_id="strongest-trend",
            category="trend",
            title="Largest observed time change",
            statement=(
                f"{column} {direction} by {abs(relative_change):.1%} when the earliest and latest "
                "20% of observed periods are compared."
            ),
            evidence=[
                f"Time column: {time_column}",
                f"Observed periods: {periods:,}",
                f"Early-period mean: {start_value:,.3f}; late-period mean: {end_value:,.3f}",
            ],
            recommendation="Validate seasonality and external events before extrapolating this historical change.",
            confidence=self._sample_confidence(periods),
            severity="info",
            sample_size=int(periods),
            metrics={
                "time_column": str(time_column),
                "value_column": column,
                "relative_change": round(float(relative_change), 6),
                "period_count": int(periods),
            },
        )

    def _largest_outlier_group(self) -> Optional[EvidenceInsight]:
        strongest = None
        for column in self._numeric_columns(exclude_identifiers=True)[:30]:
            values = pd.to_numeric(self.df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 8:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            count = int(((values < lower) | (values > upper)).sum())
            rate = count / len(values)
            candidate = (rate, count, str(column), len(values), lower, upper)
            if count and (strongest is None or candidate[0] > strongest[0]):
                strongest = candidate

        if strongest is None:
            return None

        rate, count, column, sample_size, lower, upper = strongest
        severity = "warning" if rate >= 0.05 else "info"
        return EvidenceInsight(
            insight_id="potential-outliers",
            category="distribution",
            title="Potential outliers detected",
            statement=f"{column} contains {count:,} potential IQR outliers ({rate:.1%} of valid values).",
            evidence=[
                f"Valid observations: {sample_size:,}",
                f"1.5×IQR bounds: {lower:,.3f} to {upper:,.3f}",
            ],
            recommendation=(
                "Validate these records against source data and domain rules. Do not remove them automatically; "
                "they may represent valid rare events."
            ),
            confidence=self._sample_confidence(sample_size),
            severity=severity,
            sample_size=int(sample_size),
            metrics={
                "column": column,
                "outlier_count": count,
                "outlier_rate": round(float(rate), 6),
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
            },
        )

    def _largest_category_concentration(self) -> Optional[EvidenceInsight]:
        strongest = None
        for column in self._categorical_columns()[:30]:
            values = self.df[column].dropna()
            unique_count = int(values.nunique())
            if len(values) < 10 or unique_count < 2 or unique_count > 100:
                continue
            counts = values.value_counts()
            top_value = str(counts.index[0])
            top_count = int(counts.iloc[0])
            top_share = top_count / len(values)
            candidate = (top_share, str(column), top_value, top_count, len(values), unique_count)
            if strongest is None or candidate[0] > strongest[0]:
                strongest = candidate

        if strongest is None or strongest[0] < 0.50:
            return None

        top_share, column, top_value, top_count, sample_size, unique_count = strongest
        return EvidenceInsight(
            insight_id="category-concentration",
            category="segment",
            title="Category concentration",
            statement=f"{top_value!r} represents {top_share:.1%} of non-missing values in {column}.",
            evidence=[
                f"Top-category rows: {top_count:,} of {sample_size:,}",
                f"Distinct non-missing categories: {unique_count:,}",
            ],
            recommendation="Check whether this imbalance is expected; use stratified evaluation if this is a target.",
            confidence=self._sample_confidence(sample_size),
            severity="warning" if top_share >= 0.80 else "info",
            sample_size=int(sample_size),
            metrics={
                "column": column,
                "top_value": top_value,
                "top_count": top_count,
                "top_share": round(float(top_share), 6),
                "unique_count": unique_count,
            },
        )

    def _small_sample_warning(self) -> Optional[EvidenceInsight]:
        row_count = len(self.df)
        if row_count >= 30:
            return None
        return EvidenceInsight(
            insight_id="small-sample",
            category="limitation",
            title="Limited sample size",
            statement=f"Only {row_count:,} rows are available, so patterns may be unstable.",
            evidence=["Support tier: low because sample size is below 30 rows"],
            recommendation="Collect more observations or validate findings on an independent dataset.",
            confidence="high",
            severity="warning",
            sample_size=int(row_count),
            metrics={"row_count": int(row_count), "recommended_minimum": 30},
        )

    def _numeric_columns(self, exclude_identifiers: bool = False) -> List[Any]:
        columns = list(self.df.select_dtypes(include=[np.number]).columns)
        if not exclude_identifiers:
            return columns
        return [column for column in columns if not self._looks_like_identifier(column)]

    def _categorical_columns(self) -> List[Any]:
        return list(self.df.select_dtypes(include=["object", "string", "category", "bool"]).columns)

    def _looks_like_identifier(self, column: Any) -> bool:
        normalized = str(column).strip().lower().replace("-", "_").replace(" ", "_")
        id_name = normalized == "id" or normalized.endswith("_id") or normalized in {"index", "row_number"}
        return id_name and self.df[column].nunique(dropna=True) >= max(1, int(len(self.df) * 0.95))

    def _time_column(self) -> Optional[Any]:
        for column in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                return column
        keywords = ("date", "time", "timestamp", "datetime", "дата", "время")
        for column in self.df.columns:
            if not any(keyword in str(column).lower() for keyword in keywords):
                continue
            parsed = pd.to_datetime(self.df[column], errors="coerce")
            if parsed.notna().mean() >= 0.80:
                return column
        return None

    @staticmethod
    def _sample_confidence(sample_size: int) -> str:
        if sample_size >= 100:
            return "high"
        if sample_size >= 30:
            return "medium"
        return "low"
