import unittest
from datetime import date, datetime

from vibedash.decision_queue import build_decision_queue


class DecisionQueueTests(unittest.TestCase):
    def decision_case(self, case_id, review_date, *, status="tracking", updated_at=None):
        return {
            "id": case_id * 32,
            "status": status,
            "review_date": review_date,
            "created_at": "2026-09-01T00:00:00.000+00:00",
            "updated_at": updated_at or "2026-09-01T00:00:00.000+00:00",
        }

    def cases(self):
        return [
            self.decision_case("a", "2026-09-20"),
            self.decision_case("b", "2026-09-25"),
            self.decision_case("c", "2026-09-28"),
            self.decision_case("d", "2026-10-20"),
            self.decision_case(
                "e",
                "2026-09-10",
                status="validated",
                updated_at="2026-09-24T00:00:00.000+00:00",
            ),
        ]

    def test_active_queue_orders_cases_by_review_urgency(self):
        queue = build_decision_queue(
            reversed(self.cases()),
            view="active",
            today=date(2026, 9, 25),
        )

        self.assertEqual(
            [item["id"] for item in queue["items"]],
            ["a" * 32, "b" * 32, "c" * 32, "d" * 32],
        )
        self.assertEqual(
            [item["review_state"] for item in queue["items"]],
            ["overdue", "today", "upcoming", "scheduled"],
        )
        self.assertEqual(queue["items"][0]["timing_label"], "Overdue by 5 days")
        self.assertEqual(queue["items"][2]["timing_label"], "Due in 3 days")
        self.assertEqual(
            queue["counts"],
            {
                "active": 4,
                "overdue": 1,
                "today": 1,
                "upcoming": 1,
                "closed": 1,
                "all": 5,
            },
        )

    def test_each_queue_view_returns_only_its_cases(self):
        expected = {
            "overdue": ["a" * 32],
            "today": ["b" * 32],
            "upcoming": ["c" * 32],
            "closed": ["e" * 32],
            "all": ["a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32],
        }
        for view, identifiers in expected.items():
            with self.subTest(view=view):
                queue = build_decision_queue(
                    self.cases(),
                    view=view,
                    today=date(2026, 9, 25),
                )
                self.assertEqual(
                    [item["id"] for item in queue["items"]],
                    identifiers,
                )

    def test_invalid_view_date_and_case_are_rejected(self):
        with self.assertRaises(ValueError):
            build_decision_queue(self.cases(), view="unknown")
        with self.assertRaises(ValueError):
            build_decision_queue(self.cases(), today="2026-09-25")
        with self.assertRaises(ValueError):
            build_decision_queue(self.cases(), today=datetime(2026, 9, 25))
        with self.assertRaises(ValueError):
            build_decision_queue(["not-a-case"], today=date(2026, 9, 25))
        with self.assertRaises(ValueError):
            build_decision_queue(
                [{"status": "unknown", "review_date": "2026-09-25"}],
                today=date(2026, 9, 25),
            )
        with self.assertRaises(ValueError):
            build_decision_queue(
                [{"id": "f" * 32, "status": "tracking", "review_date": "soon"}],
                today=date(2026, 9, 25),
            )


if __name__ == "__main__":
    unittest.main()
