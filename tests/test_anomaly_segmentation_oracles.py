import unittest

import numpy as np
import pandas as pd

from vibedash.anomaly_segmentation_engine import AnomalySegmentationEngine
from vibedash.decision_brief import build_decision_brief


def _known_clusters() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    points = np.r_[
        rng.normal([-4, -4], 0.5, (60, 2)),
        rng.normal([0, 4], 0.5, (60, 2)),
        rng.normal([4, -2], 0.5, (60, 2)),
        [[15, 15], [-15, 12]],
    ]
    return pd.DataFrame(
        {
            "row_id": np.arange(len(points)),
            "feature_x": points[:, 0],
            "feature_y": points[:, 1],
        }
    )


def _independent_noise() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    return pd.DataFrame(
        {f"noise_{index}": rng.normal(size=320) for index in range(5)}
    )


class AnomalySegmentationOracleTests(unittest.TestCase):
    def test_gap_statistic_rejects_multisegment_noise(self):
        engine = AnomalySegmentationEngine(_independent_noise())

        report = engine.analyze()
        segmentation = report["segmentation"]
        insight_ids = {
            item["insight_id"] for item in engine.evidence_insights(report)
        }

        self.assertEqual(report["status"], "ok")
        self.assertEqual(segmentation["status"], "insufficient_evidence")
        self.assertEqual(segmentation["gap_selected_k"], 1)
        self.assertEqual(segmentation["quality"], "insufficient")
        self.assertEqual(segmentation["segments"], [])
        self.assertNotIn("exploratory-segmentation", insight_ids)

    def test_known_three_cluster_oracle_is_separated_and_resample_stable(self):
        report = AnomalySegmentationEngine(_known_clusters()).analyze()
        segmentation = report["segmentation"]

        self.assertEqual(segmentation["status"], "ok")
        self.assertEqual(segmentation["gap_selected_k"], 3)
        self.assertEqual(segmentation["selected_k"], 3)
        self.assertEqual(segmentation["quality"], "strong")
        self.assertGreater(segmentation["silhouette_score"], 0.85)
        self.assertGreater(segmentation["stability_score"], 0.95)
        self.assertEqual(sum(item["size"] for item in segmentation["segments"]), 182)

    def test_affine_unit_changes_preserve_oracle_decisions(self):
        original = _known_clusters()
        transformed = original.copy()
        transformed["feature_x"] = transformed["feature_x"] * -7.0 + 125.0
        transformed["feature_y"] = transformed["feature_y"] * 0.125 - 9.0

        baseline = AnomalySegmentationEngine(original).analyze()
        changed = AnomalySegmentationEngine(transformed).analyze()

        self.assertEqual(
            {
                item["row_index"]
                for item in baseline["anomaly_detection"]["top_anomalies"][:2]
            },
            {
                item["row_index"]
                for item in changed["anomaly_detection"]["top_anomalies"][:2]
            },
        )
        self.assertEqual(
            baseline["segmentation"]["selected_k"],
            changed["segmentation"]["selected_k"],
        )
        self.assertAlmostEqual(
            baseline["segmentation"]["silhouette_score"],
            changed["segmentation"]["silhouette_score"],
            places=6,
        )

    def test_row_order_does_not_change_strong_structural_conclusion(self):
        original = _known_clusters()
        shuffled = original.sample(frac=1.0, random_state=991)

        baseline = AnomalySegmentationEngine(original).analyze()
        reordered = AnomalySegmentationEngine(shuffled).analyze()

        self.assertEqual(reordered["segmentation"]["status"], "ok")
        self.assertEqual(reordered["segmentation"]["selected_k"], 3)
        self.assertEqual(reordered["segmentation"]["quality"], "strong")
        self.assertEqual(
            {
                item["row_index"]
                for item in baseline["anomaly_detection"]["top_anomalies"][:2]
            },
            {
                item["row_index"]
                for item in reordered["anomaly_detection"]["top_anomalies"][:2]
            },
        )

    def test_anomaly_priority_is_explicitly_exploratory(self):
        pattern_report = AnomalySegmentationEngine(_known_clusters()).analyze()

        brief = build_decision_brief(
            {
                "statistical_validation": {"tests": []},
                "pattern_analysis": pattern_report,
                "insights": [],
            },
            {"status": "ready", "status_label": "Ready", "checks": []},
        )

        priority = brief["priorities"][0]
        self.assertEqual(priority["category"], "anomaly-review")
        self.assertEqual(priority["confidence"], "exploratory")
        self.assertIn("Candidate policy: fixed review budget", priority["evidence"])


if __name__ == "__main__":
    unittest.main()
