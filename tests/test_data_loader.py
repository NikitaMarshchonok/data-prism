import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data_loader import is_supported_data_file, load_data


class DataLoaderTests(unittest.TestCase):
    def test_supported_extensions_are_case_insensitive(self):
        for filename in (
            "data.CSV",
            "data.tsv",
            "data.XLSX",
            "data.xls",
            "data.JSON",
            "data.parquet",
        ):
            with self.subTest(filename=filename):
                self.assertTrue(is_supported_data_file(filename))

        self.assertFalse(is_supported_data_file("data.exe"))
        self.assertFalse(is_supported_data_file(""))

    def test_csv_loading_stops_at_row_limit(self):
        with TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "large.csv"
            csv_path.write_text("value\n1\n2\n3\n", encoding="utf-8")

            data, truncated = load_data(str(csv_path), max_rows=2)

        self.assertTrue(truncated)
        self.assertEqual(data["value"].tolist(), [1, 2])

    def test_json_dataset_is_supported(self):
        with TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "dataset.json"
            json_path.write_text('[{"value": 1}, {"value": 2}]', encoding="utf-8")

            data, truncated = load_data(str(json_path))

        self.assertFalse(truncated)
        self.assertEqual(data["value"].tolist(), [1, 2])

    def test_unsupported_file_returns_no_dataset(self):
        with TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "dataset.txt"
            file_path.write_text("value\n1\n", encoding="utf-8")

            data, truncated = load_data(str(file_path))

        self.assertIsNone(data)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
