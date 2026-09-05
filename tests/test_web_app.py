import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from werkzeug.datastructures import FileStorage

import web_app


class WebUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.upload_folder = Path(self.temporary_directory.name) / "uploads"
        self.report_folder = Path(self.temporary_directory.name) / "reports"
        self.baseline_folder = Path(self.temporary_directory.name) / "baselines"
        self.upload_folder.mkdir()
        self.report_folder.mkdir()
        self.previous_config = {
            "TESTING": web_app.app.config.get("TESTING"),
            "UPLOAD_FOLDER": web_app.app.config["UPLOAD_FOLDER"],
            "REPORT_FOLDER": web_app.app.config["REPORT_FOLDER"],
            "BASELINE_FOLDER": web_app.app.config["BASELINE_FOLDER"],
            "MAX_CONTENT_LENGTH": web_app.app.config["MAX_CONTENT_LENGTH"],
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(self.upload_folder),
            REPORT_FOLDER=str(self.report_folder),
            BASELINE_FOLDER=str(self.baseline_folder),
        )

    def tearDown(self):
        web_app.app.config.update(self.previous_config)
        self.temporary_directory.cleanup()

    def test_upload_name_is_sanitized_and_dataset_is_normalized(self):
        uploaded_file = FileStorage(
            stream=BytesIO(b"value,label\n1,a\n2,b\n"),
            filename="../../unsafe.csv",
        )

        data, truncated, dataset_id, dataset_filename = web_app.save_uploaded_dataset(
            uploaded_file
        )

        self.assertFalse(truncated)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(dataset_filename, f"{dataset_id}.csv")
        self.assertTrue((self.upload_folder / dataset_filename).exists())
        self.assertFalse((Path(self.temporary_directory.name) / "unsafe.csv").exists())
        source_files = [
            path.name
            for path in self.upload_folder.iterdir()
            if path.name != dataset_filename
        ]
        self.assertEqual(source_files, [f"{dataset_id}_unsafe.csv"])

    def test_unsupported_upload_is_rejected_without_writing(self):
        uploaded_file = FileStorage(
            stream=BytesIO(b"not a dataset"),
            filename="payload.exe",
        )

        with self.assertRaisesRegex(ValueError, "Неподдерживаемый формат"):
            web_app.save_uploaded_dataset(uploaded_file)

        self.assertEqual(list(self.upload_folder.iterdir()), [])

    @patch("web_app.generate_report")
    def test_upload_route_stores_dataset_reference_in_session(self, generate_report):
        with web_app.app.test_client() as client:
            response = client.post(
                "/",
                data={
                    "datafile": (
                        BytesIO(b"value\n1\n2\n3\n"),
                        "dataset.csv",
                    )
                },
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 302)
            with client.session_transaction() as current_session:
                dataset_filename = current_session["dataset_filename"]
                report_filename = current_session["report_filename"]

        self.assertTrue((self.upload_folder / dataset_filename).exists())
        self.assertTrue(report_filename.endswith(".html"))
        generate_report.assert_called_once()

    def test_dashboard_requires_an_uploaded_dataset(self):
        with web_app.app.test_client() as client:
            response = client.get("/dashboard")

        self.assertEqual(response.status_code, 400)

    @patch("src.data_loader.load_data")
    def test_dashboard_can_persist_current_dataset_as_baseline(self, load_data):
        load_data.return_value = (
            pd.DataFrame(
                {
                    "value": list(range(20)),
                    "region": ["north", "south"] * 10,
                }
            ),
            False,
        )
        with web_app.app.test_client() as client:
            with client.session_transaction() as current_session:
                current_session["dataset_filename"] = "current.csv"
                current_session["dataset_name"] = "current.csv"

            response = client.post(
                "/dashboard",
                data={"dashboard_action": "set_drift_baseline"},
            )

            self.assertEqual(response.status_code, 302)
            with client.session_transaction() as current_session:
                baseline_filename = current_session["baseline_profile_filename"]

        baseline_path = self.baseline_folder / baseline_filename
        self.assertTrue(baseline_path.exists())
        self.assertIn('"profile_version": 1', baseline_path.read_text(encoding="utf-8"))

    @patch("src.dashboard_generator.generate_dashboard_data")
    @patch("src.data_loader.load_data")
    def test_dashboard_renders_drift_against_saved_baseline(
        self,
        load_data,
        generate_dashboard_data,
    ):
        baseline_data = pd.DataFrame({"amount": list(range(100))})
        current_data = pd.DataFrame({"amount": list(range(1000, 1100))})
        baseline_filename = f"{'a' * 32}.json"
        web_app.save_baseline_profile(
            web_app.create_baseline_profile(baseline_data, dataset_name="reference.csv"),
            web_app.baseline_profile_path(baseline_filename),
        )
        load_data.return_value = current_data, False
        generate_dashboard_data.return_value = ({}, [], [], "summary", {}, "", None)

        with web_app.app.test_client() as client:
            with client.session_transaction() as current_session:
                current_session["dataset_filename"] = "current.csv"
                current_session["baseline_profile_filename"] = baseline_filename

            response = client.get("/dashboard")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Мониторинг data drift", html)
        self.assertIn("Статус: critical", html)
        self.assertIn("amount", html)

    def test_upload_larger_than_configured_limit_is_rejected(self):
        web_app.app.config["MAX_CONTENT_LENGTH"] = 256

        with web_app.app.test_client() as client:
            response = client.post(
                "/",
                data={"datafile": (BytesIO(b"x" * 1024), "dataset.csv")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
