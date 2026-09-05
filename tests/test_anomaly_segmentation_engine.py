import json
import unittest

import numpy as np
import pandas as pd

from vibedash.ai_analyzer import DataScienceAI
from vibedash.anomaly_segmentation_engine import AnomalySegmentationEngine


def _clustered_dataset():
    rng = np.random.default_rng(42)
    points = np.r_[
        rng.normal([-4, -4], 0.5, (60, 2)),
        rng.normal([0, 4], 0.5, (60, 2)),
        rng.normal([4, -2], 0.5, (60, 2)),
        [[15, 15], [-15, 12]],
    ]
    return pd.DataFrame(
        {
            "row_id": range(len(points)),
            "feature_x": points[:, 0],
            "feature_y": points[:, 1],
        }
    )


class AnomalySegmentationEngineTests(unittest.TestCase):
    def test_extreme_rows_are_prioritized_with_feature_reasons(self):
        report = AnomalySegmentationEngine(_clustered_dataset()).analyze()
        anomaly = report["anomaly_detection"]

        self.assertEqual(report["status"], "ok")
        self.assertNotIn("row_id", report["features"])
        self.assertGreaterEqual(anomaly["flagged_count"], 2)
        top_positions = {item["row_position"] for item in anomaly["top_anomalies"][:2]}
        self.assertEqual(top_positions, {180, 181})
        self.assertTrue(anomaly["top_anomalies"][0]["reasons"])
        self.assertGreaterEqual(anomaly["top_anomalies"][0]["anomaly_score"], 0.99)
        json.dumps(report)

    def test_segment_count_is_selected_by_quality_and_is_stable(self):
        report = AnomalySegmentationEngine(_clustered_dataset()).analyze()
        segmentation = report["segmentation"]

        self.assertEqual(segmentation["status"], "ok")
        self.assertEqual(segmentation["selected_k"], 3)
        self.assertEqual(segmentation["quality"], "strong")
        self.assertGreater(segmentation["silhouette_score"], 0.8)
        self.assertGreater(segmentation["stability_score"], 0.9)
        self.assertEqual(sum(item["size"] for item in segmentation["segments"]), 182)
        self.assertEqual(len(segmentation["candidate_scores"]), 5)

    def test_single_feature_keeps_anomaly_scan_but_skips_segmentation(self):
        data = pd.DataFrame({"value": np.r_[np.linspace(0, 1, 39), 20]})

        report = AnomalySegmentationEngine(data).analyze()

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["anomaly_detection"]["status"], "ok")
        self.assertEqual(report["segmentation"]["status"], "insufficient_data")

    def test_small_dataset_returns_explanatory_result(self):
        data = pd.DataFrame({"x": range(10), "y": range(10)})

        report = AnomalySegmentationEngine(data).analyze()

        self.assertEqual(report["status"], "insufficient_data")
        self.assertIn("At least 20 rows", report["summary"])
        self.assertEqual(report["anomaly_detection"]["top_anomalies"], [])

    def test_chat_clustering_uses_validated_pattern_report(self):
        result = DataScienceAI(_clustered_dataset()).analyze_question(
            "find clusters and segments"
        )

        self.assertIn("pattern_analysis", result)
        self.assertIn("silhouette", result["answer"])
        self.assertIn("exploratory segments", result["answer"])
        self.assertTrue(any(item.get("type") == "evidence" for item in result["insights"]))


if __name__ == "__main__":
    unittest.main()
