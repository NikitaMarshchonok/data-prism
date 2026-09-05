import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.drift_store import DriftStore


def make_report(status="stable", baseline_created_at="2026-01-01T00:00:00+00:00"):
    return {
        "status": status,
        "baseline_created_at": baseline_created_at,
        "summary": f"Report status: {status}",
        "feature_drift": [],
        "schema_changes": {
            "missing_columns": [],
            "new_columns": [],
            "type_changes": [],
        },
    }


class DriftStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "drift.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_stable_run_is_stored_without_alert(self):
        store = DriftStore(self.database_path, "scope-a")

        result = store.record_run(
            make_report("stable"),
            batch_id="batch-1",
            dataset_name="current.csv",
            checked_at="2026-01-02T00:00:00+00:00",
        )

        self.assertTrue(result["created"])
        self.assertFalse(result["alert_created"])
        self.assertEqual(store.list_runs()[0]["dataset_name"], "current.csv")
        self.assertEqual(store.list_alerts(), [])

    def test_critical_run_creates_acknowledgeable_alert(self):
        store = DriftStore(self.database_path, "scope-a")
        store.record_run(
            make_report("critical"),
            batch_id="batch-1",
            checked_at="2026-01-02T00:00:00+00:00",
        )

        alert = store.list_alerts()[0]
        acknowledged = store.acknowledge_alert(
            alert["id"],
            acknowledged_at="2026-01-03T00:00:00+00:00",
        )

        self.assertTrue(acknowledged)
        self.assertEqual(store.list_alerts(), [])
        all_alerts = store.list_alerts(unacknowledged_only=False)
        self.assertEqual(all_alerts[0]["acknowledged_at"], "2026-01-03T00:00:00+00:00")

    def test_duplicate_baseline_batch_pair_is_not_recorded_twice(self):
        store = DriftStore(self.database_path, "scope-a")
        first = store.record_run(make_report("warning"), batch_id="batch-1")
        duplicate = store.record_run(make_report("warning"), batch_id="batch-1")

        self.assertTrue(first["created"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(len(store.list_runs()), 1)
        self.assertEqual(len(store.list_alerts()), 1)

    def test_history_and_alerts_are_isolated_by_scope(self):
        first_store = DriftStore(self.database_path, "scope-a")
        second_store = DriftStore(self.database_path, "scope-b")
        recorded = first_store.record_run(make_report("critical"), batch_id="batch-1")
        first_alert = first_store.list_alerts()[0]

        self.assertEqual(len(first_store.list_runs()), 1)
        self.assertEqual(second_store.list_runs(), [])
        self.assertEqual(second_store.list_alerts(), [])
        self.assertFalse(second_store.acknowledge_alert(first_alert["id"]))
        self.assertEqual(len(first_store.list_alerts()), 1)
        self.assertEqual(first_store.get_run(recorded["run_id"])["status"], "critical")
        self.assertIsNone(second_store.get_run(recorded["run_id"]))

    def test_retention_removes_old_runs_and_their_alerts(self):
        store = DriftStore(self.database_path, "scope-a", retention=2)
        for index in range(3):
            store.record_run(
                make_report("warning"),
                batch_id=f"batch-{index}",
                checked_at=f"2026-01-0{index + 1}T00:00:00+00:00",
            )

        runs = store.list_runs()
        alerts = store.list_alerts()

        self.assertEqual([run["batch_id"] for run in runs], ["batch-2", "batch-1"])
        self.assertEqual(len(alerts), 2)

    def test_invalid_report_status_is_rejected(self):
        store = DriftStore(self.database_path, "scope-a")

        with self.assertRaisesRegex(ValueError, "unsupported status"):
            store.record_run(make_report("unknown"), batch_id="batch-1")


if __name__ == "__main__":
    unittest.main()
