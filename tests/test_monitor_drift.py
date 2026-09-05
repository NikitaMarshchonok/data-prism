import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from monitor_drift import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_THRESHOLD_REACHED,
    create_baseline,
    _load_stable_dataset,
    load_job_config,
    main,
    run_monitoring_job,
)


class MonitorDriftCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.baseline_data = self.root / "baseline.csv"
        self.current_data = self.root / "current.csv"
        self.baseline_profile = self.root / "baseline.json"
        self.history_database = self.root / "history.sqlite3"
        self.config_path = self.root / "monitoring.json"
        _write_csv(self.baseline_data, range(100))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_create_baseline_and_run_stable_job(self):
        baseline_result = create_baseline(self.baseline_data, self.baseline_profile)
        _write_csv(self.current_data, range(100))
        self._write_config(fail_on="critical")

        payload, exit_code = run_monitoring_job(self.config_path)

        self.assertEqual(baseline_result["row_count"], 100)
        self.assertFalse(baseline_result["stores_raw_rows"])
        self.assertEqual(exit_code, EXIT_OK)
        self.assertEqual(payload["status"], "stable")
        self.assertFalse(payload["threshold_reached"])

    def test_baseline_output_cannot_overwrite_source_dataset(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            create_baseline(self.baseline_data, self.baseline_data)

    def test_warning_threshold_returns_machine_readable_exit_code(self):
        create_baseline(self.baseline_data, self.baseline_profile)
        _write_csv(self.current_data, range(1000, 1100))
        self._write_config(fail_on="warning")

        first, first_exit = run_monitoring_job(self.config_path, batch_id="batch-1")
        duplicate, duplicate_exit = run_monitoring_job(
            self.config_path,
            batch_id="batch-1",
        )

        self.assertEqual(first["status"], "critical")
        self.assertEqual(first_exit, EXIT_THRESHOLD_REACHED)
        self.assertTrue(first["run"]["alert_created"])
        self.assertFalse(duplicate["run"]["created"])
        self.assertEqual(duplicate_exit, EXIT_THRESHOLD_REACHED)

    def test_fail_on_never_records_critical_run_without_failing_job(self):
        create_baseline(self.baseline_data, self.baseline_profile)
        _write_csv(self.current_data, range(1000, 1100))
        self._write_config(fail_on="never")

        payload, exit_code = run_monitoring_job(self.config_path)

        self.assertEqual(payload["status"], "critical")
        self.assertEqual(exit_code, EXIT_OK)
        self.assertTrue(payload["run"]["alert_created"])

    def test_main_emits_one_json_document_to_stdout(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "create-baseline",
                    "--data",
                    str(self.baseline_data),
                    "--output",
                    str(self.baseline_profile),
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, EXIT_OK)
        self.assertEqual(payload["command"], "create-baseline")

    def test_main_reports_execution_errors_as_json(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "run",
                    "--config",
                    str(self.root / "missing.json"),
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, EXIT_ERROR)
        self.assertEqual(payload["command"], "run")
        self.assertEqual(payload["error"]["type"], "ValueError")

    def test_config_rejects_unknown_fields_and_resolves_relative_paths(self):
        create_baseline(self.baseline_data, self.baseline_profile)
        _write_csv(self.current_data, range(100))
        self._write_config(extra={"typo_field": True})

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            load_job_config(self.config_path)

        self._write_config()
        loaded = load_job_config(self.config_path)
        self.assertEqual(loaded["current_data"], self.current_data.resolve())
        self.assertEqual(loaded["history_database"], self.history_database.resolve())

    def test_history_database_cannot_overwrite_job_inputs(self):
        create_baseline(self.baseline_data, self.baseline_profile)
        _write_csv(self.current_data, range(100))
        self._write_config(extra={"history_database": self.current_data.name})

        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            load_job_config(self.config_path)

    @patch("monitor_drift._file_sha256", side_effect=["before", "after"])
    def test_dataset_change_during_read_is_rejected(self, _checksum):
        with self.assertRaisesRegex(RuntimeError, "changed while"):
            _load_stable_dataset(self.baseline_data)

    def _write_config(self, *, fail_on="critical", extra=None):
        config = {
            "job_id": "sales-production-drift",
            "job_name": "daily-sales-drift",
            "baseline_profile": self.baseline_profile.name,
            "current_data": self.current_data.name,
            "history_database": self.history_database.name,
            "fail_on": fail_on,
            "retention": 10,
        }
        config.update(extra or {})
        self.config_path.write_text(json.dumps(config), encoding="utf-8")


def _write_csv(path, values):
    path.write_text(
        "value\n" + "\n".join(str(value) for value in values) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
