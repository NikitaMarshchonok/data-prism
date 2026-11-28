"""Files from different fields and tools go through the real upload path."""

import json
import os
import time
import unittest
import unittest.mock
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import web_app


def _sports_cp1251():
    rng = np.random.default_rng(1)
    lines = ["Дата;Игрок;Команда;Голы;Минуты;Рейтинг"]
    for index in range(60):
        lines.append(
            f"2025-{1 + index % 12:02d}-{1 + index % 27:02d};Игрок {index % 9};"
            f"{'Спартак' if index % 2 else 'Зенит'};{rng.integers(0, 4)};{rng.integers(30, 91)};"
            f"{rng.uniform(5, 9):.1f}".replace(".", ",")
        )
    return "\r\n".join(lines).encode("cp1251"), "матчи.csv", 6


def _shop_cp1255():
    rng = np.random.default_rng(2)
    lines = ["תאריך,סניף,הכנסות,לקוחות"]
    for index in range(60):
        lines.append(
            f"2025-{1 + index % 12:02d}-{1 + index % 27:02d},{'חיפה' if index % 2 else 'תל אביב'},"
            f"\"{rng.integers(1000, 9000):,}.50\",{rng.integers(10, 90)}"
        )
    return "\r\n".join(lines).encode("cp1255"), "sales.csv", 4


def _lab_cp1252():
    rng = np.random.default_rng(3)
    lines = ["Probe;Gruppe;Temperatur °C;Konzentration µg/l"]
    for index in range(60):
        concentration = f"{rng.uniform(900, 2500):,.1f}".replace(",", " ").replace(".", ",")
        lines.append(f"P{index};{'Kontrolle' if index % 2 else 'Behandlung'};{rng.uniform(18, 25):.2f};{concentration}".replace(".", ","))
    return "\r\n".join(lines).encode("cp1252"), "labor.csv", 4


def _statistics_utf16():
    rng = np.random.default_rng(4)
    lines = ["Region\tYear\tPopulation\tUnemployment"]
    for index in range(60):
        lines.append(f"R{index % 6}\t{2000 + index % 20}\t{rng.integers(10_000, 90_000)}\t{rng.uniform(2, 12):.2f}")
    return "\r\n".join(lines).encode("utf-16"), "stats.txt", 4


def _museum_xlsx():
    rng = np.random.default_rng(5)
    buffer = BytesIO()
    body = pd.DataFrame({
        "Museum": [f"Museum {index % 4}" for index in range(60)],
        "Date": pd.date_range("2025-01-01", periods=60),
        "Visitors": rng.integers(80, 900, 60),
        "Ticket price": rng.choice([10.0, 12.5, 15.0], 60),
    })
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([["Visitor report"]]).to_excel(writer, sheet_name="Visits", header=False, index=False)
        body.to_excel(writer, sheet_name="Visits", startrow=2, index=False)
    return buffer.getvalue(), "museums.xlsx", 4


def _sensors_parquet():
    rng = np.random.default_rng(6)
    buffer = BytesIO()
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=60, freq="h"),
        "sensor": [f"S{index % 3}" for index in range(60)],
        "temperature": rng.normal(20, 2, 60),
        "humidity": rng.uniform(30, 70, 60),
    }).to_parquet(buffer)
    return buffer.getvalue(), "sensors.parquet", 4


FIELDS = {
    "sports": _sports_cp1251,
    "commerce": _shop_cp1255,
    "science": _lab_cp1252,
    "public statistics": _statistics_utf16,
    "culture": _museum_xlsx,
    "sensors": _sensors_parquet,
}


class UniversalUploadRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        keys = ("UPLOAD_FOLDER", "VIBEDASH_JOB_STORE_PATH", "VIBEDASH_RETENTION_HOURS", "MAX_ROWS_PREVIEW")
        self.previous_config = {key: web_app.app.config[key] for key in keys}
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(root / "uploads"),
            VIBEDASH_JOB_STORE_PATH=str(root / "jobs.sqlite3"),
            VIBEDASH_RETENTION_HOURS=24,
        )
        Path(web_app.app.config["UPLOAD_FOLDER"]).mkdir()

    def tearDown(self):
        web_app.app.config.update(self.previous_config)
        self.temporary_directory.cleanup()

    def _run(self, client, data, filename):
        preflight = client.post("/vibedash/readiness", data={"datafile": (BytesIO(data), filename)})
        self.assertEqual(preflight.status_code, 200, preflight.get_data(as_text=True))
        payload = preflight.get_json()
        self.assertTrue(payload["readiness"]["analysis_allowed"])
        proposal = payload["semantic_contract"]
        proposal["confirmed"] = True
        queued = client.post("/vibedash/jobs", data={
            "prompt": "What stands out in this table?",
            "semantic_contract": json.dumps(proposal),
            "datafile": (BytesIO(data), filename),
        })
        self.assertEqual(queued.status_code, 202, queued.get_data(as_text=True))
        status_url = queued.get_json()["status_url"]
        status = {}
        for _ in range(400):
            status = client.get(status_url).get_json()
            if status["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        self.assertEqual(status["status"], "completed")
        manifest = client.get(status["manifest_url"]).get_json()
        return payload, manifest

    def test_tables_from_six_fields_are_read_and_analysed(self):
        with unittest.mock.patch.dict(os.environ, {"DATA_PRISM_STATE_DIR": self.temporary_directory.name}):
            with web_app.app.test_client() as client:
                for field_name, build in FIELDS.items():
                    data, filename, columns = build()
                    with self.subTest(field=field_name):
                        payload, manifest = self._run(client, data, filename)
                        self.assertEqual(payload["input"]["columns"], columns)
                        self.assertEqual(payload["input"]["warnings"], [] if not filename.endswith(".xlsx") else payload["input"]["warnings"])
                        self.assertEqual(manifest["dataset"]["column_count"], columns)
                        self.assertEqual(manifest["dataset"]["read"]["format"], payload["input"]["format"])
                        self.assertEqual(manifest["dataset"]["row_selection"]["method"], "all_rows")

    def test_detected_encoding_and_numbers_are_reported_before_analysis(self):
        data, filename, _ = _sports_cp1251()
        with web_app.app.test_client() as client:
            payload = client.post("/vibedash/readiness", data={"datafile": (BytesIO(data), filename)}).get_json()
        self.assertEqual(payload["input"]["encoding"], "cp1251")
        self.assertEqual(payload["input"]["delimiter"], "semicolon")
        self.assertEqual(payload["input"]["number_formats"], {"Рейтинг": "decimal comma"})
        names = [column["name"] for column in payload["semantic_contract"]["columns"]]
        self.assertEqual(names, ["Дата", "Игрок", "Команда", "Голы", "Минуты", "Рейтинг"])

    def test_unsupported_files_are_rejected_with_the_accepted_formats(self):
        with web_app.app.test_client() as client:
            response = client.post("/vibedash/readiness", data={"datafile": (BytesIO(b"%PDF-1.7"), "report.pdf")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Excel", response.get_json()["error"])

    def test_large_dated_tables_keep_the_most_recent_rows(self):
        web_app.app.config["MAX_ROWS_PREVIEW"] = 40
        rng = np.random.default_rng(8)
        rows = ["date,region,revenue"] + [
            f"{day.date()},{'north' if index % 2 else 'south'},{rng.integers(100, 900)}"
            for index, day in enumerate(pd.date_range("2025-01-01", periods=100))
        ]
        with unittest.mock.patch.dict(os.environ, {"DATA_PRISM_STATE_DIR": self.temporary_directory.name}):
            with web_app.app.test_client() as client:
                _, manifest = self._run(client, "\n".join(rows).encode(), "daily.csv")
        selection = manifest["dataset"]["row_selection"]
        self.assertEqual(selection["method"], "latest_rows_by_time")
        self.assertEqual(selection["time_column"], "date")
        self.assertEqual(selection["kept_from"][:10], "2025-03-02")
        self.assertEqual(selection["kept_to"][:10], "2025-04-10")
        self.assertTrue(manifest["dataset"]["truncated"])


class MetricExpressionTests(unittest.TestCase):
    def test_columns_with_spaces_units_and_commas_get_working_kpis(self):
        from vibedash.generator_bridge import _evaluate_metric
        from vibedash.spec import parse_prompt_to_viz_spec

        rng = np.random.default_rng(9)
        frame = pd.DataFrame({
            "Temperatur °C": rng.normal(22, 2, 40),
            "Время на поле, мин": rng.integers(30, 91, 40),
        })
        spec = parse_prompt_to_viz_spec("Average values", list(frame.columns), dataframe=frame)
        expressions = [metric.expr for metric in spec.metrics]
        self.assertIn("mean(`Temperatur °C`)", expressions)
        self.assertIn("mean(`Время на поле, мин`)", expressions)
        for expression in expressions:
            _evaluate_metric(frame, expression)


if __name__ == "__main__":
    unittest.main()
