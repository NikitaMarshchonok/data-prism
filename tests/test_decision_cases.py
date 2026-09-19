import sqlite3
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from vibedash.decision_cases import (
    DecisionCaseCapacityError,
    DecisionCaseConflictError,
    DecisionCaseStore,
    MAX_DECISION_CASES,
)


def evidence_snapshot(job_id):
    return {
        "contract": "decision-case-source-v1",
        "analysis_job_id": job_id,
        "dataset_sha256": "a" * 64,
        "priority": {
            "number": 1,
            "title": "Investigate activation decline",
            "finding": "Activation declined by 8 percentage points.",
            "confidence": "high",
            "evidence": ["Activation: 42% versus 50% baseline."],
        },
    }


class DecisionCaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "jobs.sqlite3"
        self.store = DecisionCaseStore(self.database_path)
        self.scope_id = uuid.uuid4().hex
        self.job_id = uuid.uuid4().hex

    def tearDown(self):
        self.temporary_directory.cleanup()

    def create_case(self, job_id=None, **overrides):
        effective_job_id = job_id or self.job_id
        values = {
            "priority": 1,
            "owner": "Growth lead",
            "decision": "Test a shorter onboarding path.",
            "success_metric": "Activation rate",
            "target_outcome": "Increase activation from 42% to 47%.",
            "review_date": "2026-10-15",
            "evidence_snapshot": evidence_snapshot(effective_job_id),
        }
        values.update(overrides)
        return self.store.create(self.scope_id, effective_job_id, **values)

    def test_case_round_trips_and_is_isolated_by_scope(self):
        decision_case = self.create_case()

        self.assertEqual(decision_case["status"], "tracking")
        self.assertEqual(decision_case["evidence_snapshot"]["contract"], "decision-case-source-v1")
        self.assertEqual(
            [item["id"] for item in self.store.list_for_scope(self.scope_id)],
            [decision_case["id"]],
        )
        self.assertIsNone(self.store.get(decision_case["id"], uuid.uuid4().hex))

    def test_same_priority_cannot_be_tracked_twice(self):
        self.create_case()

        with self.assertRaises(DecisionCaseConflictError):
            self.create_case()

        existing = self.store.find_for_job_priority(self.scope_id, self.job_id, 1)
        self.assertIsNotNone(existing)

    def test_terminal_status_requires_and_records_actual_outcome(self):
        decision_case = self.create_case()

        with self.assertRaises(ValueError):
            self.store.update_outcome(
                decision_case["id"],
                self.scope_id,
                status="validated",
                actual_outcome="",
            )

        updated = self.store.update_outcome(
            decision_case["id"],
            self.scope_id,
            status="validated",
            actual_outcome="Activation reached 48% after four weeks.",
        )
        self.assertEqual(updated["status"], "validated")
        self.assertIsNotNone(updated["resolved_at"])

        reopened = self.store.update_outcome(
            decision_case["id"],
            self.scope_id,
            status="tracking",
            actual_outcome="Interim result only.",
        )
        self.assertIsNone(reopened["resolved_at"])

    def test_capacity_and_input_bounds_are_enforced(self):
        self.create_case(max_cases_per_scope=1)

        with self.assertRaises(DecisionCaseCapacityError):
            self.store.create(
                self.scope_id,
                uuid.uuid4().hex,
                priority=2,
                owner="Revenue lead",
                decision="Review packaging.",
                success_metric="Expansion revenue",
                target_outcome="Increase by 5%.",
                review_date="2026-11-01",
                evidence_snapshot=evidence_snapshot(uuid.uuid4().hex),
                max_cases_per_scope=1,
            )
        with self.assertRaises(ValueError):
            self.create_case(owner="x" * 121)
        with self.assertRaises(ValueError):
            self.create_case(review_date="tomorrow")

    def test_paginated_cases_are_scoped_ordered_and_report_sentinel(self):
        older = self.create_case()
        newer = self.create_case(job_id=uuid.uuid4().hex)
        other = self.store.create(
            uuid.uuid4().hex,
            uuid.uuid4().hex,
            priority=1,
            owner="Revenue lead",
            decision="Review packaging.",
            success_metric="Expansion revenue",
            target_outcome="Increase by 5%.",
            review_date="2026-11-01",
            evidence_snapshot=evidence_snapshot(uuid.uuid4().hex),
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE decision_cases SET created_at = ? WHERE id IN (?, ?, ?)",
                (
                    "2026-01-01T00:00:00.000+00:00",
                    older["id"],
                    newer["id"],
                    other["id"],
                ),
            )

        first_page, has_more = self.store.list_for_scope_page(
            self.scope_id,
            limit=1,
        )
        self.assertEqual([case["id"] for case in first_page], [newer["id"]])
        self.assertTrue(has_more)

        final_page, has_more = self.store.list_for_scope_page(
            self.scope_id,
            limit=2,
        )
        self.assertEqual(
            [case["id"] for case in final_page],
            [newer["id"], older["id"]],
        )
        self.assertFalse(has_more)
        self.assertNotIn(other["id"], [case["id"] for case in final_page])

    def test_paginated_cases_validate_limit_boundaries(self):
        self.assertEqual(
            self.store.list_for_scope_page(
                self.scope_id,
                limit=MAX_DECISION_CASES,
            ),
            ([], False),
        )
        for invalid_limit in (0, MAX_DECISION_CASES + 1, True, 1.5):
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(ValueError):
                    self.store.list_for_scope_page(
                        self.scope_id,
                        limit=invalid_limit,
                    )
        with self.assertRaises(ValueError):
            self.store.list_for_scope_page("not-a-scope", limit=1)

    def test_only_expired_terminal_cases_are_purged(self):
        terminal = self.create_case()
        active = self.store.create(
            self.scope_id,
            uuid.uuid4().hex,
            priority=2,
            owner="Revenue lead",
            decision="Keep monitoring expansion revenue.",
            success_metric="Expansion revenue",
            target_outcome="Remain above baseline.",
            review_date="2026-11-01",
            evidence_snapshot=evidence_snapshot(uuid.uuid4().hex),
        )
        self.store.update_outcome(
            terminal["id"],
            self.scope_id,
            status="invalidated",
            actual_outcome="Activation remained at 42%.",
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE decision_cases SET updated_at = ? WHERE id IN (?, ?)",
                ("2000-01-01T00:00:00.000+00:00", terminal["id"], active["id"]),
            )

        self.assertEqual(self.store.purge_terminal(30), 1)
        self.assertIsNone(self.store.get(terminal["id"]))
        self.assertIsNotNone(self.store.get(active["id"]))


if __name__ == "__main__":
    unittest.main()
