import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data_analyzer import analyze_data_quality
from src.report_generator import generate_report


class AnalyzeDataQualityTests(unittest.TestCase):
    def test_clean_dataset_gets_full_score(self):
        data = pd.DataFrame(
            {
                "value": [10, 11, 12, 13],
                "category": ["a", "b", "c", "d"],
            }
        )

        result = analyze_data_quality(data)

        self.assertEqual(result["score"], 100)
        self.assertEqual(result["status"], "Отличное")
        self.assertEqual(result["missing_cells"], 0)
        self.assertEqual(result["duplicate_rows"], 0)
        self.assertEqual(result["constant_columns"], [])
        self.assertEqual(result["outliers"], {})
        self.assertEqual(
            result["recommendations"],
            ["Критичных проблем качества не обнаружено."],
        )

    def test_detects_common_quality_problems(self):
        data = pd.DataFrame(
            {
                "value": [1, 1, 2, 2, 100, None],
                "category": ["x", "x", "x", "x", "x", "x"],
            }
        )

        result = analyze_data_quality(data)

        self.assertEqual(result["missing_cells"], 1)
        self.assertEqual(result["duplicate_rows"], 2)
        self.assertEqual(result["constant_columns"], ["category"])
        self.assertEqual(result["outliers"], {"value": 1})
        self.assertEqual(len(result["recommendations"]), 4)
        self.assertLess(result["score"], 100)

    def test_empty_dataset_returns_explanatory_result(self):
        result = analyze_data_quality(pd.DataFrame())

        self.assertEqual(result["score"], 0)
        self.assertEqual(result["status"], "Нет данных")
        self.assertEqual(len(result["recommendations"]), 1)

    def test_quality_audit_is_rendered_in_report(self):
        data_quality = analyze_data_quality(
            pd.DataFrame({"value": [1, 2, 3], "constant": ["x", "x", "x"]})
        )
        template_path = Path(__file__).resolve().parents[1] / "templates"

        with TemporaryDirectory() as temp_directory:
            output_file = Path(temp_directory) / "report.html"
            generate_report(
                stats={},
                interactive_charts={},
                data_quality=data_quality,
                template_path=str(template_path),
                output_file=str(output_file),
            )
            report_html = output_file.read_text(encoding="utf-8")

        self.assertIn("Качество данных:", report_html)
        self.assertIn("Постоянные столбцы:</b> constant", report_html)
        self.assertIn("Удалите постоянные столбцы", report_html)


if __name__ == "__main__":
    unittest.main()
