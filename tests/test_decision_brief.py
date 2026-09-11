import json
import unittest

from vibedash.decision_brief import build_decision_brief


class DecisionBriefTests(unittest.TestCase):
    def test_brief_prioritizes_validated_signal_anomaly_and_readiness(self):
        readiness = {
            "status": "ready_with_warnings",
            "status_label": "Ready with warnings",
            "checks": [
                {
                    "check_id": "missingness",
                    "title": "Missing data",
                    "status": "warning",
                    "description": "Values are missing.",
                    "evidence": ["Missing cells: 10"],
                    "recommendation": "Review missingness.",
                }
            ],
        }
        dashboard = {
            "statistical_validation": {
                "tests": [
                    {
                        "title": "Revenue by plan",
                        "significant": True,
                        "adjusted_p_value": 0.001,
                        "sample_size": 120,
                        "effect_size": {"name": "Hedges' g", "value": 1.1},
                        "confidence_interval": {"lower": 0.8, "upper": 1.4},
                        "interpretation": "The groups differ after FDR correction.",
                        "limitations": ["Groups may differ on unobserved factors."],
                    }
                ]
            },
            "pattern_analysis": {
                "anomaly_detection": {
                    "status": "ok",
                    "flagged_count": 3,
                    "evaluated_rows": 120,
                    "summary": "Three rows require review.",
                    "top_anomalies": [
                        {
                            "row_index": "17",
                            "reasons": [
                                {"feature": "revenue", "robust_deviation": 4.2}
                            ],
                        }
                    ],
                }
            },
            "insights": [],
        }

        brief = build_decision_brief(dashboard, readiness)

        self.assertEqual(brief["contract"], "decision-brief-v1")
        self.assertEqual(brief["status"], "review_required")
        self.assertEqual(brief["priority_count"], 3)
        self.assertEqual(
            [item["category"] for item in brief["priorities"]],
            ["validated-signal", "anomaly-review", "data-readiness"],
        )
        self.assertEqual([item["priority"] for item in brief["priorities"]], [1, 2, 3])
        json.dumps(brief)

    def test_brief_never_promotes_inconclusive_statistical_test(self):
        brief = build_decision_brief(
            {
                "statistical_validation": {
                    "tests": [{"title": "Noise", "significant": False}]
                },
                "pattern_analysis": {"status": "insufficient_data"},
                "insights": [],
            },
            {
                "status": "ready",
                "status_label": "Ready",
                "checks": [],
            },
        )

        self.assertEqual(brief["status"], "insufficient_evidence")
        self.assertEqual(brief["priorities"], [])


if __name__ == "__main__":
    unittest.main()
