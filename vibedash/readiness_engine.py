"""Deterministic dataset readiness checks for VibeDash analyses."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


READINESS_CONTRACT = "dataset-readiness-v1"
MIN_ANALYSIS_ROWS = 8
RECOMMENDED_ANALYSIS_ROWS = 30

_SENSITIVE_NAME_PATTERN = re.compile(
    r"(^|_)(email|e_mail|phone|mobile|ssn|social_security|passport|"
    r"national_id|credit_card|card_number|date_of_birth|dob)($|_)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReadinessCheck:
    """One bounded readiness result with evidence and remediation."""

    check_id: str
    title: str
    status: str
    description: str
    evidence: List[str]
    recommendation: str
    penalty: int = 0
    metrics: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["metrics"] = result["metrics"] or {}
        return result


class DatasetReadinessEngine:
    """Assess whether a table can support defensible automated analysis."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df

    def assess(self) -> Dict[str, Any]:
        checks = [
            self._shape_check(),
            self._column_identity_check(),
            self._usable_signal_check(),
            self._missingness_check(),
            self._duplicate_row_check(),
            self._finite_numeric_check(),
            self._sensitive_schema_check(),
            self._high_cardinality_check(),
        ]
        penalties = sum(check.penalty for check in checks)
        score = max(0, 100 - penalties)
        blocked = sum(check.status == "blocked" for check in checks)
        warnings = sum(check.status == "warning" for check in checks)
        if blocked:
            status = "blocked"
            label = "Blocked"
        elif warnings:
            status = "ready_with_warnings"
            label = "Ready with warnings"
        else:
            status = "ready"
            label = "Ready"

        return {
            "contract": READINESS_CONTRACT,
            "status": status,
            "status_label": label,
            "score": score,
            "summary": self._summary(status, score, blocked, warnings),
            "counts": {
                "blocked": blocked,
                "warnings": warnings,
                "passed": sum(check.status == "passed" for check in checks),
                "informational": sum(check.status == "info" for check in checks),
            },
            "dataset": {
                "row_count": int(len(self.df)),
                "column_count": int(len(self.df.columns)),
                "numeric_column_count": int(
                    len(self.df.select_dtypes(include=[np.number]).columns)
                ),
                "categorical_column_count": int(
                    len(
                        self.df.select_dtypes(
                            include=["object", "string", "category", "bool"]
                        ).columns
                    )
                ),
            },
            "checks": [check.to_dict() for check in checks],
            "analysis_allowed": not bool(blocked),
        }

    def _shape_check(self) -> ReadinessCheck:
        rows, columns = self.df.shape
        if rows == 0 or columns == 0:
            return ReadinessCheck(
                "dataset-shape",
                "Dataset coverage",
                "blocked",
                "The dataset has no analyzable rows or columns.",
                [f"Observed shape: {rows} rows × {columns} columns"],
                "Upload a table with a header and at least eight data rows.",
                100,
                {"row_count": rows, "column_count": columns},
            )
        if rows < MIN_ANALYSIS_ROWS:
            return ReadinessCheck(
                "dataset-shape",
                "Dataset coverage",
                "blocked",
                "The sample is too small for the automated evidence contracts.",
                [f"Observed rows: {rows}", f"Required rows: {MIN_ANALYSIS_ROWS}"],
                "Provide at least eight observations or review the records manually.",
                45,
                {"row_count": rows, "minimum_rows": MIN_ANALYSIS_ROWS},
            )
        if rows < RECOMMENDED_ANALYSIS_ROWS:
            return ReadinessCheck(
                "dataset-shape",
                "Dataset coverage",
                "warning",
                "The sample supports limited exploration but weak statistical conclusions.",
                [
                    f"Observed rows: {rows}",
                    f"Recommended rows: {RECOMMENDED_ANALYSIS_ROWS} or more",
                ],
                "Collect more observations and treat detected patterns as exploratory.",
                15,
                {"row_count": rows, "recommended_rows": RECOMMENDED_ANALYSIS_ROWS},
            )
        return ReadinessCheck(
            "dataset-shape",
            "Dataset coverage",
            "passed",
            "The table has enough rows for the supported exploratory checks.",
            [f"Observed rows: {rows}", f"Observed columns: {columns}"],
            "Confirm that the uploaded period and population match the decision question.",
            metrics={"row_count": rows, "column_count": columns},
        )

    def _column_identity_check(self) -> ReadinessCheck:
        normalized = [str(column).strip() for column in self.df.columns]
        duplicate_names = sorted(
            {name for name in normalized if normalized.count(name) > 1}
        )
        unnamed = [
            name
            for name in normalized
            if not name or name.lower().startswith("unnamed:")
        ]
        if duplicate_names:
            return ReadinessCheck(
                "column-identity",
                "Column identity",
                "blocked",
                "Duplicate column names make calculations ambiguous.",
                ["Duplicate names: " + ", ".join(duplicate_names[:10])],
                "Rename duplicate columns so every field has a unique meaning.",
                35,
                {"duplicate_name_count": len(duplicate_names)},
            )
        if unnamed:
            return ReadinessCheck(
                "column-identity",
                "Column identity",
                "warning",
                "Placeholder column names reduce interpretability.",
                ["Placeholder names: " + ", ".join(unnamed[:10])],
                "Rename or remove index-like columns before sharing the analysis.",
                8,
                {"placeholder_name_count": len(unnamed)},
            )
        return ReadinessCheck(
            "column-identity",
            "Column identity",
            "passed",
            "Every column has a distinct, non-placeholder name.",
            [f"Unique column names: {len(normalized)}"],
            "Keep names stable across future uploads to support drift monitoring.",
            metrics={"unique_column_name_count": len(set(normalized))},
        )

    def _usable_signal_check(self) -> ReadinessCheck:
        varying = [
            str(column)
            for column, series in self._iter_columns()
            if series.nunique(dropna=True) >= 2
        ]
        constant = [
            str(column)
            for column, series in self._iter_columns()
            if series.nunique(dropna=True) <= 1
        ]
        if not varying:
            return ReadinessCheck(
                "usable-signal",
                "Usable analytical signal",
                "blocked",
                "No column varies enough to support comparison or trend analysis.",
                [f"Constant or empty columns: {len(constant)}"],
                "Provide at least one field with two or more observed values.",
                60,
                {"varying_columns": 0, "constant_columns": len(constant)},
            )
        if len(varying) == 1:
            return ReadinessCheck(
                "usable-signal",
                "Usable analytical signal",
                "warning",
                "Only one field varies, so relationships and group comparisons are limited.",
                [f"Varying column: {varying[0]}", f"Constant columns: {len(constant)}"],
                "Add relevant dimensions or outcomes if the decision requires comparison.",
                12,
                {"varying_columns": 1, "constant_columns": len(constant)},
            )
        status = "warning" if constant else "passed"
        penalty = min(12, 3 + len(constant)) if constant else 0
        return ReadinessCheck(
            "usable-signal",
            "Usable analytical signal",
            status,
            (
                "Some fields are constant and cannot explain variation."
                if constant
                else "Multiple varying fields support comparisons and relationships."
            ),
            [
                f"Varying columns: {len(varying)}",
                f"Constant or empty columns: {len(constant)}",
            ],
            (
                "Remove constant fields unless they are required for schema compatibility."
                if constant
                else "Confirm the business meaning of fields before acting on relationships."
            ),
            penalty,
            {"varying_columns": len(varying), "constant_columns": len(constant)},
        )

    def _missingness_check(self) -> ReadinessCheck:
        total_cells = int(self.df.shape[0] * self.df.shape[1])
        missing_cells = int(self.df.isna().sum().sum())
        missing_rate = missing_cells / total_cells if total_cells else 1.0
        empty_columns = [
            str(column)
            for column, series in self._iter_columns()
            if series.isna().all()
        ]
        metrics = {
            "missing_cells": missing_cells,
            "missing_rate": round(float(missing_rate), 6),
            "empty_column_count": len(empty_columns),
        }
        evidence = [
            f"Missing cells: {missing_cells} of {total_cells} ({missing_rate:.1%})",
            f"Fully empty columns: {len(empty_columns)}",
        ]
        if missing_rate >= 0.50:
            return ReadinessCheck(
                "missingness",
                "Missing data",
                "blocked",
                "At least half of the table is missing, making automatic conclusions unsafe.",
                evidence,
                "Correct the export scope or provide a documented missing-value strategy.",
                40,
                metrics,
            )
        if missing_rate >= 0.10 or empty_columns:
            return ReadinessCheck(
                "missingness",
                "Missing data",
                "warning",
                "Missing values can change aggregates, group comparisons, and model coverage.",
                evidence,
                "Review missingness by field and document any removal or imputation rule.",
                18 if missing_rate >= 0.20 else 10,
                metrics,
            )
        if missing_cells:
            return ReadinessCheck(
                "missingness",
                "Missing data",
                "warning",
                "A small amount of missing data is present.",
                evidence,
                "Confirm that missing values are expected before interpreting affected metrics.",
                4,
                metrics,
            )
        return ReadinessCheck(
            "missingness",
            "Missing data",
            "passed",
            "No missing cells were detected.",
            evidence,
            "Continue monitoring missingness on future data refreshes.",
            metrics=metrics,
        )

    def _duplicate_row_check(self) -> ReadinessCheck:
        rows = len(self.df)
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_rate = duplicate_rows / rows if rows else 0.0
        metrics = {
            "duplicate_rows": duplicate_rows,
            "duplicate_rate": round(float(duplicate_rate), 6),
        }
        evidence = [
            f"Duplicate rows: {duplicate_rows} of {rows} ({duplicate_rate:.1%})"
        ]
        if duplicate_rate >= 0.25:
            return ReadinessCheck(
                "duplicate-rows",
                "Duplicate observations",
                "blocked",
                "Duplicate observations are likely to materially distort totals and frequencies.",
                evidence,
                "Confirm the row grain and deduplicate using a documented business key.",
                30,
                metrics,
            )
        if duplicate_rows:
            return ReadinessCheck(
                "duplicate-rows",
                "Duplicate observations",
                "warning",
                "Repeated rows may overstate aggregates or group evidence.",
                evidence,
                "Verify whether repetitions are valid events before removing them.",
                12 if duplicate_rate >= 0.05 else 5,
                metrics,
            )
        return ReadinessCheck(
            "duplicate-rows",
            "Duplicate observations",
            "passed",
            "No fully duplicated rows were detected.",
            evidence,
            "Confirm the intended row grain when interpreting totals.",
            metrics=metrics,
        )

    def _finite_numeric_check(self) -> ReadinessCheck:
        numeric = self.df.select_dtypes(include=[np.number])
        infinite_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if not numeric.empty else 0
        if infinite_count:
            return ReadinessCheck(
                "finite-numeric-values",
                "Finite numeric values",
                "warning",
                "Infinite numeric values cannot be interpreted as measured quantities.",
                [f"Infinite values: {infinite_count}"],
                "Replace infinities with a documented missing or bounded value before analysis.",
                10,
                {"infinite_value_count": infinite_count},
            )
        return ReadinessCheck(
            "finite-numeric-values",
            "Finite numeric values",
            "passed",
            "No infinite numeric values were detected.",
            ["Infinite values: 0"],
            "No remediation is required.",
            metrics={"infinite_value_count": 0},
        )

    def _sensitive_schema_check(self) -> ReadinessCheck:
        sensitive = [
            str(column)
            for column in self.df.columns
            if _SENSITIVE_NAME_PATTERN.search(_normalize_name(column))
        ]
        if sensitive:
            return ReadinessCheck(
                "sensitive-schema",
                "Potentially sensitive fields",
                "warning",
                "Column names indicate that the upload may contain personal or regulated data.",
                ["Potentially sensitive columns: " + ", ".join(sensitive[:10])],
                "Remove or tokenize unnecessary identifiers and confirm an approved retention policy.",
                15,
                {"potentially_sensitive_column_count": len(sensitive)},
            )
        return ReadinessCheck(
            "sensitive-schema",
            "Potentially sensitive fields",
            "passed",
            "No common sensitive field names were detected in the schema.",
            ["Potentially sensitive column names: 0"],
            "Schema checks cannot inspect policy context; upload only authorized data.",
            metrics={"potentially_sensitive_column_count": 0},
        )

    def _high_cardinality_check(self) -> ReadinessCheck:
        candidates = []
        for column, series in self._iter_columns():
            if not (
                pd.api.types.is_object_dtype(series.dtype)
                or pd.api.types.is_string_dtype(series.dtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                continue
            if _looks_like_datetime(series):
                continue
            observed = int(series.notna().sum())
            unique = int(series.nunique(dropna=True))
            ratio = unique / observed if observed else 0.0
            if unique > 50 and ratio > 0.50:
                candidates.append(str(column))
        if candidates:
            return ReadinessCheck(
                "high-cardinality-fields",
                "High-cardinality dimensions",
                "warning",
                "Some text fields behave like identifiers or free text and may fragment groups.",
                ["High-cardinality fields: " + ", ".join(candidates[:10])],
                "Exclude identifiers from grouping and model features unless their role is intentional.",
                min(12, 5 + len(candidates)),
                {"high_cardinality_column_count": len(candidates)},
            )
        return ReadinessCheck(
            "high-cardinality-fields",
            "High-cardinality dimensions",
            "passed",
            "No high-cardinality text dimensions require automatic exclusion.",
            ["High-cardinality text fields: 0"],
            "Review domain identifiers manually because names and distributions are only heuristics.",
            metrics={"high_cardinality_column_count": 0},
        )

    def _iter_columns(self):
        for position, column in enumerate(self.df.columns):
            yield column, self.df.iloc[:, position]

    @staticmethod
    def _summary(status: str, score: int, blocked: int, warnings: int) -> str:
        if status == "blocked":
            return (
                f"Analysis blocked at readiness score {score}/100: {blocked} blocking "
                f"issue(s) and {warnings} warning(s) require review."
            )
        if status == "ready_with_warnings":
            return (
                f"Analysis can continue at readiness score {score}/100 with "
                f"{warnings} documented warning(s)."
            )
        return f"Analysis is ready at readiness score {score}/100 with no detected warnings."


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _looks_like_datetime(series: pd.Series) -> bool:
    """Keep valid time dimensions out of the identifier heuristic."""
    observed = series.dropna().astype(str).head(100)
    if len(observed) < 8:
        return False
    parsed = pd.to_datetime(observed, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.90)
