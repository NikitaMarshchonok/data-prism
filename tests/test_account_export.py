import json
import math
import unittest
from datetime import datetime, timezone

from vibedash.account_export import (
    ACCOUNT_EXPORT_CONTRACT,
    MAX_ACCOUNT_EXPORT_BYTES,
    MAX_ACCOUNT_EXPORT_CASES,
    MAX_ACCOUNT_EXPORT_JOBS,
    AccountExportError,
    AccountExportTooLargeError,
    build_account_export,
    serialize_account_export,
)


class AccountExportTests(unittest.TestCase):
    def setUp(self):
        self.job_id = "a" * 32
        self.case_id = "b" * 32
        self.generated_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    def build(self, jobs=None, cases=None, **kwargs):
        return build_account_export(
            {"id": "secret", "scope_id": "secret", "email": "owner@example.com"},
            jobs if jobs is not None else [],
            jobs_truncated=False,
            decision_cases=cases if cases is not None else [],
            decisions_truncated=False,
            generated_at=self.generated_at,
            **kwargs,
        )

    def test_allowlist_excludes_raw_payload_manifest_and_snapshot_content(self):
        payload = self.build(
            jobs=[
                {
                    "id": self.job_id,
                    "status": "completed",
                    "payload": {"prompt": "PAYLOAD_SECRET", "filename": "FILE_SECRET"},
                    "manifest": {
                        "analysis_contract": "vibedash-evidence-v2",
                        "dataset": {
                            "content_sha256": "c" * 64,
                            "source_name": "SOURCE_SECRET",
                            "schema_preview": [{"name": "COLUMN_SECRET"}],
                        },
                        "specification": {"title": "TITLE_SECRET"},
                    },
                }
            ],
            cases=[
                {
                    "id": self.case_id,
                    "job_id": self.job_id,
                    "priority": 1,
                    "status": "tracking",
                    "owner": "Owner",
                    "decision": "Decision",
                    "evidence_snapshot": {
                        "analysis_title": "TITLE_SECRET",
                        "priority": {"number": 1, "finding": "SNAPSHOT_SECRET"},
                    },
                }
            ],
        )
        serialized = serialize_account_export(payload).decode()
        for marker in ("PAYLOAD_SECRET", "FILE_SECRET", "SOURCE_SECRET", "COLUMN_SECRET", "TITLE_SECRET", "SNAPSHOT_SECRET"):
            self.assertNotIn(marker, serialized)
        self.assertNotIn("id", payload["account"])
        self.assertNotIn("scope_id", serialized)

    def test_normal_and_comparison_manifest_aggregates(self):
        normal = {
            "id": self.job_id,
            "manifest": {
                "analysis_contract": "vibedash-evidence-v2",
                "dataset": {"content_sha256": "a" * 64, "source_rows": 5, "column_count": 2},
                "evidence": {"insight_count": 2, "readiness": {"status": "ready"}},
            },
        }
        comparison = {
            "id": "d" * 32,
            "payload": {"analysis_kind": "period_comparison"},
            "manifest": {
                "analysis_contract": "vibedash-period-comparison-v1",
                "inputs": {"baseline": {"filename": "secret.csv", "content_sha256": "b" * 64}},
                "comparison": {"result_sha256": "c" * 64, "counts": {"tested_metrics": 3}},
            },
        }
        result = self.build(jobs=[normal, comparison])
        self.assertEqual(result["jobs"][0]["manifest"]["dataset"]["source_rows"], 5)
        self.assertNotIn("filename", result["jobs"][1]["manifest"]["inputs"]["baseline"])
        self.assertEqual(result["jobs"][1]["manifest"]["comparison"]["counts"]["tested_metrics"], 3)

    def test_deterministic_generated_at_and_compact_sorting(self):
        first = serialize_account_export(self.build())
        second = serialize_account_export(self.build())
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["contract"], ACCOUNT_EXPORT_CONTRACT)
        self.assertNotIn(b", ", first)
        self.assertNotIn(b": ", first)

    def test_malformed_and_nan_fail_closed(self):
        result = self.build(jobs=[{"id": self.job_id, "status": "\ud800", "manifest": {"dataset": {"source_rows": math.nan}}}])
        self.assertNotIn("status", result["jobs"][0])
        self.assertNotIn("source_rows", result["jobs"][0].get("manifest", {}).get("dataset", {}))
        with self.assertRaises(AccountExportError):
            serialize_account_export({"bad": math.nan})

    def test_corrupt_decision_case_dates_are_omitted(self):
        result = self.build(cases=[{
            "id": self.case_id,
            "review_date": "not-a-date",
            "created_at": "2026-01-01T00:00:00Z",
        }])
        self.assertNotIn("review_date", result["decision_cases"][0])

    def test_corrupt_records_cannot_inject_bounded_enum_values(self):
        result = self.build(
            jobs=[{
                "id": "A" * 32,
                "status": "corrupt_status",
                "error_code": "raw_internal_error",
                "payload": {"analysis_kind": "attacker_kind"},
                "manifest": {
                    "analysis_contract": "attacker_contract",
                    "evidence": {
                        "readiness": {"contract": "attacker_readiness", "status": "attacker_status"},
                        "decision_brief": {"contract": "attacker_brief", "status": "attacker_status"},
                    },
                },
            }],
            cases=[{
                "id": "B" * 32,
                "job_id": "C" * 32,
                "status": "corrupt_case_status",
                "evidence_snapshot": {
                    "contract": "attacker_case_contract",
                    "analysis_contract": "attacker_contract",
                },
            }],
        )
        encoded = serialize_account_export(result).decode()
        for marker in (
            "corrupt_status", "raw_internal_error", "attacker_kind", "attacker_contract",
            "attacker_readiness", "attacker_brief", "attacker_status", "corrupt_case_status",
            "attacker_case_contract",
        ):
            self.assertNotIn(marker, encoded)
        self.assertNotIn("id", result["jobs"][0])
        self.assertNotIn("id", result["decision_cases"][0])

    def test_collection_and_byte_bounds(self):
        jobs = [{"id": f"{index:032x}"} for index in range(MAX_ACCOUNT_EXPORT_JOBS + 1)]
        cases = [{"id": f"{index + 1:032x}", "priority": 1} for index in range(MAX_ACCOUNT_EXPORT_CASES + 1)]
        result = self.build(jobs=jobs, cases=cases)
        self.assertEqual(len(result["jobs"]), MAX_ACCOUNT_EXPORT_JOBS)
        self.assertEqual(len(result["decision_cases"]), MAX_ACCOUNT_EXPORT_CASES)
        self.assertTrue(result["jobs_truncated"])
        self.assertTrue(result["decisions_truncated"])
        with self.assertRaises(AccountExportTooLargeError):
            serialize_account_export({"x": "x"}, max_bytes=1)
        self.assertLessEqual(len(serialize_account_export(result)), MAX_ACCOUNT_EXPORT_BYTES)


if __name__ == "__main__":
    unittest.main()
