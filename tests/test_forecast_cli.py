from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import evaluate_forecast
from tests.test_forecasting import frame


class ForecastCLITests(unittest.TestCase):
    def run_cli(self, path, *extra):
        stream = StringIO()
        with redirect_stdout(stream):
            code = evaluate_forecast.main(["--csv", str(path), "--target", "demand",
                                           "--time", "date", "--horizon", "7", *extra])
        return code, json.loads(stream.getvalue())

    def test_local_input_is_unchanged_and_complete_report_is_strict_json(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "daily.csv"
            frame().to_csv(source, index=False)
            before = source.read_bytes()
            code, report = self.run_cli(source)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [source])
        self.assertEqual(code, 0)
        self.assertEqual(report["contract"], "daily-forecast-v1")
        self.assertEqual(len(report["future_forecast"]), 7)
        json.dumps(report, allow_nan=False)

    def test_invalid_csv_headers_paths_and_encoding_do_not_echo_private_values(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "PRIVATE_PATH.csv"
            for raw in (b"date,date\nPRIVATE_VALUE,2\n", b"\xff\xfe", b""):
                source.write_bytes(raw)
                code, report = self.run_cli(source)
                self.assertEqual(code, 1)
                self.assertNotIn("PRIVATE", json.dumps(report))
            code, report = self.run_cli(Path(directory) / "PRIVATE_MISSING.csv")
            self.assertEqual(code, 1)
            self.assertNotIn("PRIVATE", json.dumps(report))

    def test_row_limit_is_a_failure_not_truncated_history(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "daily.csv"
            frame(size=5001).to_csv(source, index=False)
            code, report = self.run_cli(source)
        self.assertEqual(code, 1)
        self.assertIn("5000", report["message"])
        self.assertEqual(report["future_forecast"], [])

    def test_byte_limit_fails_before_parsing_or_training(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "daily.csv"
            source.write_text("date,demand\n2020-01-01,1\n")
            with patch("evaluate_forecast.MAX_CSV_BYTES", 10), patch("evaluate_forecast.forecast_series") as forecast:
                code, report = self.run_cli(source)
                forecast.assert_not_called()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
