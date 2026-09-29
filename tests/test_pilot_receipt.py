import json
import unittest
from datetime import datetime, timezone

from vibedash.pilot_receipt import (
    PILOT_RECEIPT_CONTRACT,
    PilotReceiptError,
    build_pilot_evidence_receipt,
    serialize_pilot_evidence_receipt,
)


class PilotEvidenceReceiptTests(unittest.TestCase):
    def outcome_summary(self):
        return {
            "contract": "decision-outcome-summary-v1",
            "retained_cases": 4,
            "tracking_cases": 1,
            "closed_cases": 3,
            "evaluated_cases": 2,
            "validated_cases": 1,
            "invalidated_cases": 1,
            "cancelled_cases": 1,
            "resolution_rate": 0.75,
            "evaluation_rate": 0.5,
            "validation_share_among_evaluated": 0.5,
            "case_id": "must-not-survive",
        }

    def value_summary(self):
        return {
            "contract": "pilot-scope-value-v1",
            "cohort_days": 30,
            "collection_installed": True,
            "completed_opted_in_analyses": 2,
            "feedback_responses": 2,
            "value_feedback_responses": 1,
            "value_feedback_response_rate_among_completed": 0.5,
            "legacy_feedback_responses_without_value_signals": 1,
            "perceived_time_saved": {
                "none": 0,
                "under_15_minutes": 0,
                "15_to_30_minutes": 1,
                "30_to_60_minutes": 0,
                "over_60_minutes": 0,
                "free_text": "must-not-survive",
            },
            "next_cycle_intent": {"yes": 1, "maybe": 0, "no": 0},
            "scope_token": "must-not-survive",
        }

    def test_receipt_allowlists_aggregate_fields(self):
        receipt = build_pilot_evidence_receipt(
            self.outcome_summary(),
            self.value_summary(),
            service_version="release-123",
            decision_retention_days=90,
            generated_at=datetime(2026, 9, 29, 8, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(receipt["contract"], PILOT_RECEIPT_CONTRACT)
        self.assertEqual(receipt["generated_at"], "2026-09-29T08:30:00+00:00")
        self.assertEqual(receipt["windows"]["decision_retention_days"], 90)
        self.assertEqual(receipt["decision_outcomes"]["evaluated_cases"], 2)
        self.assertEqual(
            receipt["value_feedback"]["perceived_time_saved"]["15_to_30_minutes"],
            1,
        )
        serialized = serialize_pilot_evidence_receipt(receipt)
        self.assertLess(len(serialized), 64 * 1024)
        decoded = json.loads(serialized)
        self.assertNotIn("scope_token", str(decoded))
        self.assertNotIn("case_id", str(decoded))
        self.assertNotIn("free_text", str(decoded))

    def test_receipt_rejects_bad_contracts_counts_ratios_and_version(self):
        cases = [
            ({**self.outcome_summary(), "contract": "unknown"}, self.value_summary(), "release-1"),
            ({**self.outcome_summary(), "retained_cases": -1}, self.value_summary(), "release-1"),
            ({**self.outcome_summary(), "resolution_rate": float("nan")}, self.value_summary(), "release-1"),
            (self.outcome_summary(), {**self.value_summary(), "cohort_days": 31}, "release-1"),
            (self.outcome_summary(), self.value_summary(), "bad\nversion"),
        ]
        for outcome, value, version in cases:
            with self.subTest(version=version, outcome=outcome, value=value):
                with self.assertRaises(PilotReceiptError):
                    build_pilot_evidence_receipt(
                        outcome,
                        value,
                        service_version=version,
                        decision_retention_days=90,
                    )

    def test_serializer_rejects_wrong_document_and_non_finite_values(self):
        with self.assertRaises(PilotReceiptError):
            serialize_pilot_evidence_receipt({"contract": "unknown"})
        with self.assertRaises(PilotReceiptError):
            serialize_pilot_evidence_receipt(
                {"contract": PILOT_RECEIPT_CONTRACT, "ratio": float("nan")}
            )


if __name__ == "__main__":
    unittest.main()
