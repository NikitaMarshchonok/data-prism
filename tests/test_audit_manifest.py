import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from vibedash.audit_manifest import build_audit_manifest


class AuditManifestTests(unittest.TestCase):
    def _build(self, dataset_path, dataframe, *, prompt="Analyze revenue"):
        return build_audit_manifest(
            dataset_path=dataset_path,
            dataframe=dataframe,
            source_row_count=len(dataframe),
            truncated=False,
            source_name="dataset.csv",
            prompt=prompt,
            demo_dataset="",
            run_id="a" * 32,
            viz_spec={
                "title": "Revenue evidence",
                "metrics": [{"title": "Total", "expr": "sum(amount)"}],
                "charts": [],
            },
            dashboard_data={
                "readiness": {
                    "contract": "dataset-readiness-v1",
                    "status": "ready_with_warnings",
                    "score": 92,
                    "counts": {"blocked": 0, "warnings": 1},
                },
                "decision_brief": {
                    "contract": "decision-brief-v1",
                    "status": "review_required",
                    "priority_count": 2,
                },
                "insights": [{"title": "Coverage"}],
                "statistical_validation": {
                    "tests": [
                        {"significant": True},
                        {"significant": False},
                    ]
                },
                "pattern_analysis": {
                    "anomaly_detection": {"top_anomalies": [{"row": 1}]}
                },
            },
            engine_version="test-release",
            retention_hours=24,
            generated_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        )

    def test_manifest_is_deterministic_and_excludes_source_values(self):
        dataframe = pd.DataFrame(
            {"amount": [10, 20], "customer": ["private-a", "private-b"]}
        )
        with TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.csv"
            dataframe.to_csv(dataset_path, index=False)
            first = self._build(dataset_path, dataframe)
            second = self._build(dataset_path, dataframe)

        self.assertEqual(first, second)
        self.assertEqual(first["manifest_version"], 2)
        self.assertEqual(first["analysis_contract"], "vibedash-evidence-v2")
        self.assertEqual(first["dataset"]["analyzed_rows"], 2)
        self.assertEqual(first["dataset"]["column_count"], 2)
        self.assertEqual(first["evidence"]["statistical_test_count"], 2)
        self.assertEqual(first["evidence"]["significant_test_count"], 1)
        self.assertEqual(first["evidence"]["anomaly_candidate_count"], 1)
        self.assertEqual(first["evidence"]["readiness"]["score"], 92)
        self.assertEqual(first["evidence"]["decision_brief"]["priority_count"], 2)
        serialized = json.dumps(first)
        self.assertNotIn("private-a", serialized)
        self.assertNotIn("private-b", serialized)
        self.assertEqual(first["request"]["prompt"], "Analyze revenue")
        self.assertEqual(first["run"]["id"], "a" * 32)
        self.assertRegex(first["dataset"]["content_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(first["dataset"]["schema_sha256"], r"^[0-9a-f]{64}$")

    def test_content_and_request_fingerprints_change_independently(self):
        dataframe = pd.DataFrame({"amount": [10, 20]})
        with TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.csv"
            dataframe.to_csv(dataset_path, index=False)
            original = self._build(dataset_path, dataframe)
            new_prompt = self._build(dataset_path, dataframe, prompt="Analyze churn")
            dataset_path.write_text("amount\n10\n30\n", encoding="utf-8")
            new_content = self._build(dataset_path, dataframe)

        self.assertEqual(
            original["dataset"]["content_sha256"],
            new_prompt["dataset"]["content_sha256"],
        )
        self.assertNotEqual(
            original["request"]["sha256"],
            new_prompt["request"]["sha256"],
        )
        self.assertNotEqual(
            original["dataset"]["content_sha256"],
            new_content["dataset"]["content_sha256"],
        )

    def test_source_rows_cannot_be_smaller_than_analyzed_rows(self):
        dataframe = pd.DataFrame({"amount": [10, 20]})
        with TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.csv"
            dataframe.to_csv(dataset_path, index=False)
            with self.assertRaises(ValueError):
                build_audit_manifest(
                    dataset_path=dataset_path,
                    dataframe=dataframe,
                    source_row_count=1,
                    truncated=False,
                    source_name="dataset.csv",
                    prompt="Analyze",
                    demo_dataset="",
                    run_id=None,
                    viz_spec={},
                    dashboard_data={},
                    engine_version="test",
                    retention_hours=24,
                )


if __name__ == "__main__":
    unittest.main()
