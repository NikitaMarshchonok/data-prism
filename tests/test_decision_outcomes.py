import unittest

from vibedash.decision_outcomes import build_decision_outcome_summary


class DecisionOutcomeSummaryTests(unittest.TestCase):
    def test_summary_uses_honest_status_denominators(self):
        summary = build_decision_outcome_summary({
            "tracking": 1,
            "validated": 2,
            "invalidated": 1,
            "cancelled": 1,
        })

        self.assertEqual(summary["retained_cases"], 5)
        self.assertEqual(summary["closed_cases"], 4)
        self.assertEqual(summary["evaluated_cases"], 3)
        self.assertEqual(summary["validated_cases"], 2)
        self.assertEqual(summary["invalidated_cases"], 1)
        self.assertEqual(summary["cancelled_cases"], 1)
        self.assertEqual(summary["tracking_cases"], 1)
        self.assertEqual(summary["resolution_rate"], 0.8)
        self.assertEqual(summary["evaluation_rate"], 0.6)
        self.assertEqual(summary["validation_share_among_evaluated"], 0.6667)

    def test_empty_summary_does_not_invent_rates(self):
        summary = build_decision_outcome_summary({})

        self.assertEqual(summary["retained_cases"], 0)
        self.assertIsNone(summary["resolution_rate"])
        self.assertIsNone(summary["evaluation_rate"])
        self.assertIsNone(summary["validation_share_among_evaluated"])

    def test_invalid_inputs_fail_closed(self):
        for invalid in (
            "validated",
            {"unknown": 1},
            {"tracking": -1},
            {"tracking": True},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_decision_outcome_summary(invalid)


if __name__ == "__main__":
    unittest.main()
