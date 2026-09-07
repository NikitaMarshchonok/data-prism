import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.demo_data import create_saas_growth_demo, main, save_demo_dataset


class DemoDataTests(unittest.TestCase):
    def test_demo_is_deterministic_and_contains_useful_analysis_signals(self):
        first = create_saas_growth_demo()
        second = create_saas_growth_demo()

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.shape, (360, 11))
        self.assertGreater(first["MonthlyRevenue"].corr(first["CustomerCount"]), 0.5)
        self.assertGreater(first["NPSScore"].isna().sum(), 0)
        self.assertGreater(first["Region"].nunique(), 2)
        self.assertNotIn("Email", first.columns)
        self.assertNotIn("Name", first.columns)

    def test_row_count_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "between 90 and 5000"):
            create_saas_growth_demo(10)

        with self.assertRaisesRegex(ValueError, "between 90 and 5000"):
            create_saas_growth_demo(5001)

    def test_demo_can_be_generated_from_the_command_line(self):
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "demo.csv"

            exit_code = main(["--output", str(output), "--rows", "120"])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())
            self.assertEqual(len(pd.read_csv(output)), 120)

    def test_save_demo_creates_parent_directories(self):
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "nested" / "demo.csv"

            saved_path = save_demo_dataset(output, row_count=90)

            self.assertEqual(saved_path, output.resolve())
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
