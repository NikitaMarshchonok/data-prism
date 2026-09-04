import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from werkzeug.datastructures import FileStorage

import web_app


class WebUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.upload_folder = Path(self.temporary_directory.name) / "uploads"
        self.report_folder = Path(self.temporary_directory.name) / "reports"
        self.upload_folder.mkdir()
        self.report_folder.mkdir()
        self.previous_config = {
            "TESTING": web_app.app.config.get("TESTING"),
            "UPLOAD_FOLDER": web_app.app.config["UPLOAD_FOLDER"],
            "REPORT_FOLDER": web_app.app.config["REPORT_FOLDER"],
            "MAX_CONTENT_LENGTH": web_app.app.config["MAX_CONTENT_LENGTH"],
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(self.upload_folder),
            REPORT_FOLDER=str(self.report_folder),
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
