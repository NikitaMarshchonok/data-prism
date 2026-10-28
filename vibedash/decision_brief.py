"""Build a concise action-oriented brief from already calculated evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping
from src.analysis_context import finding_columns, is_relevant


DECISION_BRIEF_CONTRACT = "decision-brief-v1"


def build_decision_brief(
    dashboard_data: Mapping[str, Any],
    readiness: Mapping[str, Any],
    *,
    max_priorities: int = 3,
) -> Dict[str, Any]:
    """Rank calculated findings without creating unsupported claims."""
    if max_priorities < 1:
        raise ValueError("max_priorities must be at least 1")

    priorities: List[Dict[str, Any]] = []
    statistics = dashboard_data.get("statistical_validation") or {}
    context = dashboard_data.get("analysis_context") or {}
    focused = bool(context.get("focus_columns"))
    inference_withheld = statistics.get("inference_status") == "withheld"
    if inference_withheld:
        priorities.append({
            "category": "assumption-review",
            "title": "Review the temporal sampling design before statistical inference",
            "finding": str(statistics.get("summary") or "Statistical inference was withheld."),
            "evidence": [str(note) for note in statistics.get("methodology_notes") or []],
            "risk": "Ordinary p-values and confidence intervals may be misleading when observations are dependent.",
            "action": "Establish the sampling unit and temporal dependence, then choose and validate an appropriate inferential method.",
            "confidence": "exploratory",
            "source": "statistical_validation",
        })
    significant = [
        test
        for test in statistics.get("tests") or []
        if not inference_withheld and isinstance(test, Mapping)
        and test.get("significant") and is_relevant(test, context)
    ]
    significant.sort(key=lambda test: _sortable_number(test.get("adjusted_p_value")))
    selected_signal_columns = finding_columns(significant[0]) if significant else set()
    if significant:
        test = significant[0]
        effect = test.get("effect_size") or {}
        interval = test.get("confidence_interval") or {}
        priorities.append(
            {
                "category": "validated-signal",
                "title": str(test.get("title") or "Validated statistical signal"),
                "finding": str(test.get("interpretation") or ""),
                "evidence": [
                    f"FDR-adjusted p-value: {_probability(test.get('adjusted_p_value'))}",
                    f"{effect.get('name', 'Effect size')}: {_number(effect.get('value'))}",
                    (
                        f"95% CI: [{_number(interval.get('lower'))}, "
                        f"{_number(interval.get('upper'))}] · n={int(test.get('sample_size') or 0)}"
                    ),
                ],
                "risk": _first(test.get("limitations"), "Association does not establish causality."),
                "action": "Validate the business mechanism and reproduce the signal on a future period before acting.",
                "confidence": _sample_support(int(test.get("sample_size") or 0)),
                "source": "statistical_validation",
            }
        )

    patterns = dashboard_data.get("pattern_analysis") or {}
    anomaly = patterns.get("anomaly_detection") or {}
    if anomaly.get("status") == "ok" and int(anomaly.get("flagged_count") or 0) > 0:
        top = (anomaly.get("top_anomalies") or [{}])[0]
        reasons = top.get("reasons") or []
        leading_reason = reasons[0] if reasons else {}
        priorities.append(
            {
                "category": "anomaly-review",
                "title": (
                    "Review dataset-wide unusual observations (not task-specific)"
                    if focused else "Review the highest-ranked unusual observations"
                ),
                "finding": str(anomaly.get("summary") or "Anomaly candidates were detected."),
                "evidence": [
                    f"Flagged rows: {int(anomaly.get('flagged_count') or 0)} of {int(anomaly.get('evaluated_rows') or 0)}",
                    (
                        "Candidate policy: fixed review budget"
                        if anomaly.get("candidate_policy") == "fixed_review_budget"
                        else "Candidate policy: unspecified"
                    ),
                    f"Top candidate row: {top.get('row_index', 'unavailable')}",
                    (
                        f"Leading deviation: {leading_reason.get('feature', 'unavailable')} "
                        f"({_number(leading_reason.get('robust_deviation'))} robust deviations)"
                    ),
                ],
                "risk": "Anomaly scores identify review candidates, not errors, fraud, or causal events.",
                "action": "Verify the top rows against source systems and operational event logs.",
                "confidence": "exploratory",
                "source": "pattern_analysis",
            }
        )

    warning_checks = [
        check
        for check in readiness.get("checks") or []
        if isinstance(check, Mapping) and check.get("status") in {"warning", "blocked"}
    ]
    warning_checks.sort(
        key=lambda check: (
            0 if check.get("status") == "blocked" else 1,
            -int(check.get("penalty") or 0),
        )
    )
    if warning_checks:
        check = warning_checks[0]
        priorities.append(
            {
                "category": "data-readiness",
                "title": str(check.get("title") or "Resolve a data readiness issue"),
                "finding": str(check.get("description") or ""),
                "evidence": [str(item) for item in (check.get("evidence") or [])[:3]],
                "risk": "Data limitations can change aggregates, statistical power, or model coverage.",
                "action": str(check.get("recommendation") or "Review the source data."),
                "confidence": "high",
                "source": "dataset_readiness",
            }
        )

    existing_categories = {priority["category"] for priority in priorities}
    for insight in dashboard_data.get("insights") or []:
        if not isinstance(insight, Mapping) or insight.get("insight_id") in {
            "dataset-overview",
            "data-quality",
        }:
            continue
        if insight.get("category") != "limitation" and not is_relevant(insight, context):
            continue
        if (insight.get("category") == "relationship" and selected_signal_columns
                and finding_columns(insight) == selected_signal_columns):
            continue
        category = f"evidence-{insight.get('category', 'finding')}"
        if category in existing_categories:
            continue
        priorities.append(
            {
                "category": category,
                "title": str(insight.get("title") or "Evidence-backed finding"),
                "finding": str(insight.get("statement") or ""),
                "evidence": [str(item) for item in (insight.get("evidence") or [])[:3]],
                "risk": "The finding is observational and may be affected by confounding or selection bias.",
                "action": str(insight.get("recommendation") or "Validate the finding."),
                "confidence": (
                    "exploratory"
                    if inference_withheld and insight.get("category") in {"relationship", "trend", "statistical-validation"}
                    else str(insight.get("confidence") or "unknown")
                ),
                "source": "evidence_engine",
            }
        )
        existing_categories.add(category)

    if focused:
        # Global safety warnings remain visible. Task findings precede the
        # dataset-wide anomaly ranking, which has no task-specific fit.
        priorities.sort(key=lambda item: (
            0 if item["category"] in {"assumption-review", "data-readiness", "evidence-limitation"}
            else 2 if item["category"] == "anomaly-review" else 1
        ))
    priorities = priorities[:max_priorities]
    for index, priority in enumerate(priorities, start=1):
        priority["priority"] = index

    if readiness.get("status") == "blocked":
        status = "blocked"
        headline = "Resolve data readiness issues before making a decision"
    elif priorities:
        status = "review_required"
        headline = f"{len(priorities)} evidence-backed priorities require review"
    else:
        status = "insufficient_evidence"
        headline = "No decision-ready signal was detected"

    return {
        "contract": DECISION_BRIEF_CONTRACT,
        "status": status,
        "headline": headline,
        "summary": (
            f"Dataset readiness is {str(readiness.get('status_label', 'unknown')).lower()}. "
            f"The brief ranks {len(priorities)} calculated item(s); it does not establish causality."
        ),
        "priority_count": len(priorities),
        "priorities": priorities,
        "guardrail": (
            "Confirm business definitions, source-system accuracy, and decision costs with a domain owner."
        ),
    }


def _first(values: Any, fallback: str) -> str:
    return str(values[0]) if isinstance(values, list) and values else fallback


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "unavailable"


def _probability(value: Any) -> str:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    return "<0.0001" if probability < 0.0001 else f"{probability:.4f}"


def _sample_support(sample_size: int) -> str:
    if sample_size >= 100:
        return "high"
    if sample_size >= 30:
        return "medium"
    return "low"


def _sortable_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")
