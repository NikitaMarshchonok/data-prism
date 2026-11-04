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
from src.ml_predictor import predict_target
from src.forecasting import ForecastTask, forecast_series
from src.forecast_policy import CONTRACT as FORECAST_POLICY_CONTRACT, weighted_absolute_error
from vibedash.anomaly_segmentation_engine import AnomalySegmentationEngine
from vibedash.insight_engine import EvidenceBasedInsightEngine
from vibedash.statistical_engine import StatisticalValidationEngine


QUALITY_CONTRACT_VERSION = 2
SUPPORTED_SCENARIOS = (
    "saas_known_signals",
    "null_noise_guardrail",
    "known_group_effect",
    "independent_predictive_signal",
    "daily_forecast_contract",
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
        _evaluate_independent_prediction(scenario_config["independent_predictive_signal"]),
        _evaluate_daily_forecast(scenario_config["daily_forecast_contract"]),
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
    prediction = predict_target(data, "MonthlyRevenue")

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
            "temporal-inference-withheld",
            statistics.get("status") == "inference_withheld"
            and statistics.get("inference_status") == "withheld"
            and statistics["tests"] == []
            and statistics["significant_test_count"] is None
            and correlation is not None
            and correlation["confidence"] == "exploratory",
            {"status": statistics["status"], "significant_test_count": statistics["significant_test_count"]},
            {"status": "inference_withheld", "significant_test_count": None},
            "The dated SaaS sample retains descriptive signals without manufacturing IID significance; independent-sample sensitivity is checked by the known-group-effect scenario.",
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
            "dated-random-prediction-withheld",
            prediction.get("status") == "error"
            and prediction.get("predictive_support", {}).get("status") == "not_evaluated",
            prediction.get("status"), "error",
            "Dated SaaS rows cannot silently receive random-split predictive support.",
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


def _evaluate_independent_prediction(thresholds):
    # Separate IID detection oracle: do not relabel a dated SaaS series IID
    # merely to preserve its historical random-split benchmark result.
    rng = np.random.default_rng(4104)
    feature = rng.normal(size=360)
    data = pd.DataFrame({"signal": feature, "noise": rng.normal(size=360),
                         "outcome": 5 * feature + rng.normal(scale=.2, size=360)})
    report = predict_target(data, "outcome")
    support = report.get("predictive_support", {})
    improvement = report.get("metrics", {}).get("mae_improvement", 0)
    p_value = support.get("permutation_p_value", 1)
    return _scenario("independent_predictive_signal", "Independent rows with known regression signal", data, [
        _check("known-predictive-signal-supported",
               report.get("status") == "ok" and support.get("status") == "supported_on_holdout"
               and improvement >= float(thresholds["minimum_predictive_mae_improvement"])
               and p_value <= float(thresholds["maximum_predictive_permutation_p_value"]),
               {"support_status": support.get("status"), "mae_improvement": improvement,
                "permutation_p_value": p_value}, dict(thresholds),
               "The independent known signal must beat a mean baseline on reserved rows; historical thresholds are unchanged.")])


def _evaluate_daily_forecast(thresholds):
    task = ForecastTask("demand", "date", horizon=7, seasonal_period=7)
    data = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=168),
                         "demand": 100 + .7 * np.arange(168)})
    report = forecast_series(data, task)
    changed = data.copy()
    changed.loc[changed.index[-7:], "demand"] += 10000
    changed_report = forecast_series(changed, task)
    periodic = data.assign(demand=np.tile([10., 20., 30., 70., 60., 50., 40.], 24))
    simple = forecast_series(periodic, task)
    policy = {'contract': FORECAST_POLICY_CONTRACT, 'decision': 'Plan capacity in a synthetic test',
              'underprediction_weight': 5., 'overprediction_weight': 1.,
              'minimum_improvement': .1, 'maximum_mae': 0.}
    constrained = forecast_series(data, task, decision_policy=policy)
    constrained_changed = forecast_series(changed, task, decision_policy=policy)
    weighted_oracle = weighted_absolute_error([10., 10.], [8., 13.], policy)
    holdout = report.get("holdout", {})
    improvement = holdout.get("mae_improvement")
    isolated = (report.get("status") == changed_report.get("status") == "ok"
                and report["model_comparison"] == changed_report["model_comparison"]
                and [row["prediction"] for row in holdout["predictions"]]
                == [row["prediction"] for row in changed_report["holdout"]["predictions"]])
    checks = [
        _check("forecast-known-trend", report.get("status") == "ok" and improvement is not None
               and improvement >= float(thresholds["minimum_trend_mae_improvement"]),
               improvement, dict(thresholds), "A known trend beats the validation-selected naive baseline on a future horizon."),
        _check("forecast-holdout-isolation", isolated, isolated, True,
               "Changing final holdout actuals cannot affect method selection or holdout predictions."),
        _check("forecast-baseline-retained", simple.get("status") == "ok"
               and simple.get("model_name") == "seasonal_naive"
               and simple["holdout"]["metrics"]["mae"] == 0,
               simple.get("model_name"), "seasonal_naive",
               "Perfect seasonal repetition retains the exact simple baseline rather than promoting complexity."),
        _check("forecast-no-iid-significance", report.get("status") == "ok"
               and report["predictive_support"]["statistical_significance"] == "not_assessed"
               and report["prediction_intervals"] is None,
               report.get("predictive_support"), {"statistical_significance": "not_assessed"},
               "Descriptive temporal improvement is not a permutation significance or calibrated-interval claim."),
        _check("forecast-asymmetric-loss-oracle", weighted_oracle == 6.5, weighted_oracle, 6.5,
               "An underprediction of 2 at weight 5 and overprediction of 3 at weight 1 have mean penalty 6.5, not a currency value."),
        _check("forecast-decision-limit-enforced", constrained.get('status') == 'ok'
               and constrained['business_evaluation']['status'] == 'criteria_not_met_on_holdout'
               and constrained['business_evaluation']['checks']['maximum_mae'] is False,
               constrained.get('business_evaluation'), {'maximum_mae': 0., 'status': 'criteria_not_met_on_holdout'},
               "A strong relative improvement cannot override a predeclared absolute-error limit."),
        _check("forecast-policy-holdout-isolation", constrained.get('status') == constrained_changed.get('status') == 'ok'
               and constrained['model_comparison'] == constrained_changed['model_comparison']
               and constrained['model_name'] == constrained_changed['model_name'],
               constrained.get('selection_metric'), 'mean validation weighted absolute error',
               "Predeclared weighted-error selection cannot use the final holdout to choose the method or baseline."),
    ]
    return _scenario("daily_forecast_contract", "Fixed-origin daily forecasting and holdout isolation", data, checks)


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
    pattern_engine = AnomalySegmentationEngine(data)
    patterns = pattern_engine.analyze(max_anomalies=10)
    pattern_insights = pattern_engine.evidence_insights(patterns)
    prediction = predict_target(data, "segment")
    insight_ids = [item["insight_id"] for item in insights]
    forbidden_insights = list(thresholds["forbidden_insight_ids"])
    significant_count = int(statistics["significant_test_count"])
    segmentation = patterns.get("segmentation") or {}
    anomaly = patterns.get("anomaly_detection") or {}
    anomaly_insight = next(
        (
            item
            for item in pattern_insights
            if item.get("insight_id") == "multivariate-anomaly-scan"
        ),
        {},
    )
    predictive_support = prediction.get("predictive_support", {})
    predictive_metrics = prediction.get("metrics", {})

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
        _check(
            "no-invented-segmentation",
            segmentation.get("status") == "insufficient_evidence"
            and segmentation.get("gap_selected_k") == 1
            and not any(
                item.get("insight_id") == "exploratory-segmentation"
                for item in pattern_insights
            ),
            {
                "status": segmentation.get("status"),
                "gap_selected_k": segmentation.get("gap_selected_k"),
                "insight_ids": [item.get("insight_id") for item in pattern_insights],
            },
            {
                "status": "insufficient_evidence",
                "gap_selected_k": 1,
                "excludes_insight": "exploratory-segmentation",
            },
            "Independent noise is not promoted as a defensible segment structure.",
        ),
        _check(
            "no-invented-predictive-support",
            prediction.get("status") == "ok"
            and predictive_support.get("status") == "not_supported_on_holdout",
            {
                "prediction_status": prediction.get("status"),
                "support_status": predictive_support.get("status"),
                "permutation_p_value": predictive_support.get(
                    "permutation_p_value"
                ),
                "balanced_accuracy_lift": predictive_metrics.get(
                    "balanced_accuracy_lift"
                ),
            },
            {
                "prediction_status": "ok",
                "support_status": "not_supported_on_holdout",
            },
            "Seeded independent noise is not promoted as predictive evidence beyond a naive baseline.",
        ),
        _check(
            "anomaly-candidates-remain-exploratory",
            anomaly.get("candidate_policy") == "fixed_review_budget"
            and anomaly_insight.get("confidence") == "exploratory",
            {
                "candidate_policy": anomaly.get("candidate_policy"),
                "insight_confidence": anomaly_insight.get("confidence"),
            },
            {
                "candidate_policy": "fixed_review_budget",
                "insight_confidence": "exploratory",
            },
            "Unsupervised anomaly rankings remain review candidates rather than validated findings.",
        ),
    ]
    return _scenario(
        "null_noise_guardrail",
        "Independent numeric noise with no designed relationship or segment structure",
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
