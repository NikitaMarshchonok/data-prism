"""Deterministic analytical quality evaluation for Data Prism.

The benchmark uses synthetic datasets with known properties.  It tests
behavioural contracts rather than exact floating-point snapshots so the gate
remains meaningful across supported Python and dependency versions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from src.demo_data import create_saas_growth_demo
from vibedash.anomaly_segmentation_engine import AnomalySegmentationEngine
from vibedash.insight_engine import EvidenceBasedInsightEngine
from vibedash.statistical_engine import StatisticalValidationEngine


QUALITY_CONTRACT_VERSION = 1
SUPPORTED_SCENARIOS = (
    "saas_known_signals",
    "null_noise_guardrail",
    "known_group_effect",
)


@dataclass(frozen=True)
class QualityCheck:
    """One human-readable, JSON-serializable benchmark assertion."""

    check_id: str
    passed: bool
    observed: Any
    expected: Any
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_quality_config(config: Mapping[str, Any]) -> None:
    """Validate the versioned quality-gate configuration."""
    if not isinstance(config, Mapping):
        raise ValueError("Quality configuration must be a JSON object.")
    if config.get("contract_version") != QUALITY_CONTRACT_VERSION:
        raise ValueError(
            f"contract_version must be {QUALITY_CONTRACT_VERSION}."
        )

    minimum_pass_rate = config.get("minimum_pass_rate")
    if not isinstance(minimum_pass_rate, (int, float)) or isinstance(
        minimum_pass_rate, bool
    ):
        raise ValueError("minimum_pass_rate must be a number between 0 and 1.")
    if not 0 <= float(minimum_pass_rate) <= 1:
        raise ValueError("minimum_pass_rate must be a number between 0 and 1.")

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise ValueError("scenarios must be a JSON object.")
    missing = [scenario for scenario in SUPPORTED_SCENARIOS if scenario not in scenarios]
    unknown = [scenario for scenario in scenarios if scenario not in SUPPORTED_SCENARIOS]
    if missing:
        raise ValueError("Missing quality scenarios: " + ", ".join(missing))
    if unknown:
        raise ValueError("Unknown quality scenarios: " + ", ".join(unknown))
    for scenario, thresholds in scenarios.items():
        if not isinstance(thresholds, Mapping):
            raise ValueError(f"Scenario '{scenario}' must contain a JSON object.")


def run_quality_evaluation(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run every analytical benchmark and return a stable report contract."""
    validate_quality_config(config)
    scenario_config = config["scenarios"]
    scenarios = [
        _evaluate_saas_known_signals(scenario_config["saas_known_signals"]),
        _evaluate_null_noise(scenario_config["null_noise_guardrail"]),
        _evaluate_known_group_effect(scenario_config["known_group_effect"]),
    ]

    checks = [check for scenario in scenarios for check in scenario["checks"]]
    passed_checks = sum(bool(check["passed"]) for check in checks)
    total_checks = len(checks)
    pass_rate = passed_checks / total_checks if total_checks else 0.0
    minimum_pass_rate = float(config["minimum_pass_rate"])
    passed = pass_rate >= minimum_pass_rate and all(
        scenario["status"] == "passed" for scenario in scenarios
    )
    return {
        "contract_version": QUALITY_CONTRACT_VERSION,
        "status": "passed" if passed else "failed",
        "summary": (
            f"Analytical quality gate passed {passed_checks} of {total_checks} checks "
            f"({pass_rate:.1%}); required pass rate is {minimum_pass_rate:.1%}."
        ),
        "totals": {
            "scenario_count": len(scenarios),
            "check_count": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "pass_rate": round(pass_rate, 6),
            "minimum_pass_rate": minimum_pass_rate,
        },
        "scenarios": scenarios,
        "methodology_notes": [
            "All benchmark datasets are synthetic, seeded, and contain no personal data.",
            "Known-signal scenarios measure detection; a null scenario guards against invented findings.",
            "Checks use behavioural thresholds rather than exact floating-point snapshots.",
            "This gate detects analytical regressions; it is not a claim of causal or domain validity.",
        ],
    }


def _evaluate_saas_known_signals(thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    data = create_saas_growth_demo()
    insights = EvidenceBasedInsightEngine(data).generate(max_insights=10)
    repeated_insights = EvidenceBasedInsightEngine(data).generate(max_insights=10)
    statistics = StatisticalValidationEngine(data).analyze(max_results=100)
    repeated_statistics = StatisticalValidationEngine(data).analyze(max_results=100)
    patterns = AnomalySegmentationEngine(data).analyze(max_anomalies=10)

    insight_ids = [item["insight_id"] for item in insights]
    required_insights = list(thresholds["required_insight_ids"])
    correlation = next(
        (item for item in insights if item["insight_id"] == "strongest-correlation"),
        None,
    )
    expected_pair = {str(value) for value in thresholds["correlation_pair"]}
    observed_pair = (
        {
            str(correlation["metrics"]["column_x"]),
            str(correlation["metrics"]["column_y"]),
        }
        if correlation
        else set()
    )
    absolute_correlation = (
        float(correlation["metrics"]["absolute_r"]) if correlation else 0.0
    )
    anomaly = patterns.get("anomaly_detection", {})
    top_positions = {
        int(item["row_position"]) for item in anomaly.get("top_anomalies", [])
    }
    required_incidents = {int(value) for value in thresholds["required_incident_rows"]}
    segmentation = patterns.get("segmentation", {})

    checks = [
        _check(
            "required-evidence-findings",
            set(required_insights).issubset(insight_ids),
            insight_ids,
            {"contains": required_insights},
            "The evidence engine exposes every required finding category.",
        ),
        _check(
            "known-correlation-pair",
            observed_pair == expected_pair,
            sorted(observed_pair),
            sorted(expected_pair),
            "The strongest relationship matches the designed SaaS signal.",
        ),
        _check(
            "known-correlation-strength",
            absolute_correlation >= float(thresholds["minimum_absolute_correlation"]),
            round(absolute_correlation, 6),
            {"minimum": float(thresholds["minimum_absolute_correlation"])},
            "The designed revenue/customer relationship retains practical strength.",
        ),
        _check(
            "minimum-confirmed-hypotheses",
            int(statistics["significant_test_count"])
            >= int(thresholds["minimum_significant_tests"]),
            int(statistics["significant_test_count"]),
            {"minimum": int(thresholds["minimum_significant_tests"])},
            "The statistical scan retains the expected known-signal sensitivity after FDR correction.",
        ),
        _check(
            "designed-incidents-ranked",
            required_incidents.issubset(top_positions),
            sorted(top_positions),
            {"contains": sorted(required_incidents)},
            "All deliberately injected incidents appear among the ranked anomaly candidates.",
        ),
        _check(
            "segmentation-quality",
            segmentation.get("status") == "ok"
            and float(segmentation.get("silhouette_score", 0.0))
            >= float(thresholds["minimum_silhouette_score"])
            and float(segmentation.get("stability_score", 0.0))
            >= float(thresholds["minimum_stability_score"]),
            {
                "status": segmentation.get("status"),
                "silhouette_score": segmentation.get("silhouette_score"),
                "stability_score": segmentation.get("stability_score"),
            },
            {
                "status": "ok",
                "minimum_silhouette_score": float(
                    thresholds["minimum_silhouette_score"]
                ),
                "minimum_stability_score": float(
                    thresholds["minimum_stability_score"]
                ),
            },
            "Exploratory segments remain sufficiently separated and stable.",
        ),
        _check(
            "deterministic-evidence-and-statistics",
            insights == repeated_insights and statistics == repeated_statistics,
            {
                "evidence_repeat_match": insights == repeated_insights,
                "statistics_repeat_match": statistics == repeated_statistics,
            },
            {"evidence_repeat_match": True, "statistics_repeat_match": True},
            "Repeated analysis of identical input returns identical evidence and statistics.",
        ),
    ]
    return _scenario(
        "saas_known_signals",
        "Known SaaS relationships, incidents, quality issues, and segments",
        data,
        checks,
    )


def _evaluate_null_noise(thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(thresholds.get("seed", 2026))
    row_count = int(thresholds.get("row_count", 320))
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {f"noise_{index}": rng.normal(size=row_count) for index in range(5)}
    )
    data["segment"] = rng.choice(["a", "b", "c"], size=row_count)
    insights = EvidenceBasedInsightEngine(data).generate(max_insights=10)
    statistics = StatisticalValidationEngine(data).analyze(max_results=100)
    insight_ids = [item["insight_id"] for item in insights]
    forbidden_insights = list(thresholds["forbidden_insight_ids"])
    significant_count = int(statistics["significant_test_count"])

    checks = [
        _check(
            "no-invented-strong-relationship",
            not set(forbidden_insights).intersection(insight_ids),
            insight_ids,
            {"excludes": forbidden_insights},
            "Independent noise is not promoted as a strong relationship.",
        ),
        _check(
            "false-discovery-guardrail",
            significant_count <= int(thresholds["maximum_significant_tests"]),
            significant_count,
            {"maximum": int(thresholds["maximum_significant_tests"])},
            "The FDR-controlled scan does not manufacture discoveries in seeded null data.",
        ),
        _check(
            "null-scenario-hypotheses-evaluated",
            int(statistics["candidate_test_count"])
            >= int(thresholds["minimum_candidate_tests"]),
            int(statistics["candidate_test_count"]),
            {"minimum": int(thresholds["minimum_candidate_tests"])},
            "The null guardrail evaluates enough hypotheses to be meaningful.",
        ),
    ]
    return _scenario(
        "null_noise_guardrail",
        "Independent numeric noise with no designed relationship",
        data,
        checks,
    )


def _evaluate_known_group_effect(thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(thresholds.get("seed", 314))
    rows_per_group = int(thresholds.get("rows_per_group", 120))
    shift = float(thresholds.get("designed_mean_shift", 1.5))
    rng = np.random.default_rng(seed)
    groups = np.repeat(["control", "treatment"], rows_per_group)
    data = pd.DataFrame(
        {
            "outcome": rng.normal(size=len(groups))
            + (groups == "treatment") * shift,
            "cohort": groups,
        }
    )
    statistics = StatisticalValidationEngine(data).analyze(max_results=100)
    result = next(
        (
            item
            for item in statistics["tests"]
            if item["family"] == "group-comparison"
            and set(item["columns"]) == {"outcome", "cohort"}
        ),
        None,
    )
    effect = abs(float(result["effect_size"]["value"])) if result else 0.0
    interval = result.get("confidence_interval", {}) if result else {}
    excludes_zero = bool(
        interval
        and (
            float(interval["upper"]) < 0.0
            or float(interval["lower"]) > 0.0
        )
    )

    checks = [
        _check(
            "known-group-effect-detected",
            bool(result and result["significant"]),
            {
                "result_found": result is not None,
                "significant": result.get("significant") if result else None,
            },
            {"result_found": True, "significant": True},
            "The designed group difference remains detectable after FDR correction.",
        ),
        _check(
            "known-group-effect-size",
            effect >= float(thresholds["minimum_absolute_hedges_g"]),
            round(effect, 6),
            {"minimum": float(thresholds["minimum_absolute_hedges_g"])},
            "The detected difference retains the designed practical magnitude.",
        ),
        _check(
            "known-group-confidence-interval",
            excludes_zero,
            {
                "lower": interval.get("lower"),
                "upper": interval.get("upper"),
            },
            {"excludes_zero": True},
            "The 95% confidence interval for the known effect excludes zero.",
        ),
    ]
    return _scenario(
        "known_group_effect",
        "Balanced groups with a designed outcome mean shift",
        data,
        checks,
    )


def _check(
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    description: str,
) -> Dict[str, Any]:
    return QualityCheck(
        check_id=check_id,
        passed=bool(passed),
        observed=observed,
        expected=expected,
        description=description,
    ).to_dict()


def _scenario(
    scenario_id: str,
    description: str,
    data: pd.DataFrame,
    checks: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    materialized = list(checks)
    passed = sum(bool(check["passed"]) for check in materialized)
    return {
        "scenario_id": scenario_id,
        "status": "passed" if passed == len(materialized) else "failed",
        "description": description,
        "dataset": {
            "row_count": int(len(data)),
            "column_count": int(len(data.columns)),
            "synthetic": True,
        },
        "passed_checks": passed,
        "failed_checks": len(materialized) - passed,
        "checks": materialized,
    }
