import unittest
from unittest.mock import patch

import pandas as pd

import app


class AppCliTests(unittest.TestCase):
    @patch("app.generate_report")
    @patch("app.plot_histogram")
    @patch("app.load_data")
    def test_main_unpacks_loaded_dataset(self, load_data, plot_histogram, generate_report):
        data = pd.DataFrame({"first": [1, 2], "second": [3, 4]})
        load_data.return_value = (data, False)

        app.main()

        load_data.assert_called_once_with("data/test_data.csv")
        self.assertEqual(plot_histogram.call_count, 2)
        generate_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
