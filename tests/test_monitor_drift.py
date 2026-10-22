import io
import json
import unittest
import pandas as pd
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from monitor_drift import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_THRESHOLD_REACHED,
    create_baseline,
    propose_semantics,
    _load_stable_dataset,
    load_job_config,
    main,
    run_monitoring_job,
)
from src.semantic_contract import infer_semantic_contract


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
        self.contract_path = self.root / "semantics.json"
        self.contract_path.write_text(json.dumps(infer_semantic_contract(
            pd.DataFrame({"value": range(100)}), confirmed=True,
        )), encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_create_baseline_and_run_stable_job(self):
        baseline_result = self._create_baseline()
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
        self._create_baseline()
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
        self._create_baseline()
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
                    "--semantic-contract",
                    str(self.contract_path),
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
        self._create_baseline()
        _write_csv(self.current_data, range(100))
        self._write_config(extra={"typo_field": True})

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            load_job_config(self.config_path)

        self._write_config()
        loaded = load_job_config(self.config_path)
        self.assertEqual(loaded["current_data"], self.current_data.resolve())
        self.assertEqual(loaded["history_database"], self.history_database.resolve())

    def test_history_database_cannot_overwrite_job_inputs(self):
        self._create_baseline()
        _write_csv(self.current_data, range(100))
        self._write_config(extra={"history_database": self.current_data.name})

        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            load_job_config(self.config_path)

    @patch("monitor_drift._file_sha256", side_effect=["before", "after"])
    def test_dataset_change_during_read_is_rejected(self, _checksum):
        with self.assertRaisesRegex(RuntimeError, "changed while"):
            _load_stable_dataset(self.baseline_data)

    def _create_baseline(self):
        return create_baseline(self.baseline_data, self.baseline_profile,
                               semantic_contract_path=self.contract_path)

    def test_proposal_is_unconfirmed_and_cannot_overwrite_files(self):
        path = self.root / "proposal.json"
        payload = propose_semantics(self.baseline_data, path)
        contract = json.loads(path.read_text())
        self.assertFalse(payload["confirmed"])
        self.assertFalse(contract["confirmed"])
        with self.assertRaisesRegex(ValueError, "confirm"):
            create_baseline(self.baseline_data, self.baseline_profile, semantic_contract_path=path)
        for protected in (path, self.baseline_data):
            with self.assertRaisesRegex(ValueError, "new file"):
                propose_semantics(self.baseline_data, protected)

    def test_contract_required_and_protected_from_baseline_output(self):
        with self.assertRaisesRegex(ValueError, "reviewed"):
            create_baseline(self.baseline_data, self.baseline_profile)
        with self.assertRaisesRegex(ValueError, "overwrite the semantic"):
            create_baseline(self.baseline_data, self.contract_path, semantic_contract_path=self.contract_path)

    def test_auto_deduplication_includes_baseline_revision(self):
        self._create_baseline()
        _write_csv(self.current_data, range(1000, 1100))
        self._write_config()
        first, _ = run_monitoring_job(self.config_path)
        duplicate, _ = run_monitoring_job(self.config_path)
        _write_csv(self.baseline_data, range(1000, 1100))
        self._create_baseline()
        changed, _ = run_monitoring_job(self.config_path)
        self.assertEqual(first["status"], "critical")
        self.assertFalse(duplicate["run"]["created"])
        self.assertTrue(changed["run"]["created"])
        self.assertNotEqual(first["run"]["run_id"], changed["run"]["run_id"])
        self.assertEqual(changed["status"], "stable")

    def test_current_contract_provenance_and_input_protection(self):
        self._create_baseline()
        _write_csv(self.current_data, range(100))
        self._write_config(extra={"current_semantic_contract": self.contract_path.name})
        payload, _ = run_monitoring_job(self.config_path)
        self.assertEqual(payload["report"]["semantics"]["current"], "confirmed_for_batch")
        self._write_config(extra={"current_semantic_contract": self.contract_path.name,
                                  "history_database": self.contract_path.name})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            load_job_config(self.config_path)

    def test_current_semantics_conflict_returns_json_error_without_history(self):
        self._create_baseline()
        _write_csv(self.current_data, range(100))
        contract = json.loads(self.contract_path.read_text())
        contract["columns"][0]["unit"] = "currency"
        self.contract_path.write_text(json.dumps(contract), encoding="utf-8")
        self._write_config(extra={"current_semantic_contract": self.contract_path.name})
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["run", "--config", str(self.config_path)])
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("differ from the baseline", json.loads(output.getvalue())["error"]["message"])
        self.assertFalse(self.history_database.exists())

    def test_proposal_main_emits_one_json_document(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["propose-semantics", "--data", str(self.baseline_data),
                         "--output", str(self.root / "proposal.json")])
        self.assertEqual(code, EXIT_OK)
        self.assertFalse(json.loads(output.getvalue())["confirmed"])

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
