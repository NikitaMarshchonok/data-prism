import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import evaluate_quality
from src.quality_evaluation import run_quality_evaluation, validate_quality_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "quality_gate.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class QualityEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_quality_evaluation(_config())

    def test_default_gate_passes_every_scenario_and_check(self):
        self.assertEqual(self.report["status"], "passed")
        self.assertEqual(self.report["totals"]["scenario_count"], 3)
        self.assertEqual(self.report["totals"]["failed_checks"], 0)
        self.assertEqual(self.report["totals"]["pass_rate"], 1.0)
        self.assertTrue(all(item["status"] == "passed" for item in self.report["scenarios"]))
        json.dumps(self.report)

    def test_report_includes_detection_and_false_discovery_checks(self):
        checks = {
            check["check_id"]: check
            for scenario in self.report["scenarios"]
            for check in scenario["checks"]
        }

        self.assertTrue(checks["known-correlation-pair"]["passed"])
        self.assertTrue(checks["designed-incidents-ranked"]["passed"])
        self.assertTrue(checks["known-group-effect-detected"]["passed"])
        self.assertTrue(checks["false-discovery-guardrail"]["passed"])
        self.assertEqual(checks["false-discovery-guardrail"]["observed"], 0)

    def test_tightened_threshold_fails_with_machine_readable_evidence(self):
        config = copy.deepcopy(_config())
        config["scenarios"]["saas_known_signals"][
            "minimum_absolute_correlation"
        ] = 0.999

        report = run_quality_evaluation(config)
        correlation = next(
            check
            for scenario in report["scenarios"]
            for check in scenario["checks"]
            if check["check_id"] == "known-correlation-strength"
        )

        self.assertEqual(report["status"], "failed")
        self.assertFalse(correlation["passed"])
        self.assertLess(correlation["observed"], 0.999)

    def test_configuration_rejects_unknown_scenarios(self):
        config = _config()
        config["scenarios"]["unreviewed_benchmark"] = {}

        with self.assertRaisesRegex(ValueError, "Unknown quality scenarios"):
            validate_quality_config(config)

    def test_cli_writes_the_same_json_report_it_prints(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quality-report.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = evaluate_quality.main(
                    ["--config", str(CONFIG_PATH), "--output", str(output)]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
