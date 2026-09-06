import os
import unittest
from unittest.mock import patch

import pandas as pd

from src.ai_summary import generate_ai_summary_openai


class AiSummaryTests(unittest.TestCase):
    def test_missing_api_key_returns_an_explanation_without_creating_client(self):
        data = pd.DataFrame({"value": [1, 2, 3]})

        with patch.dict(os.environ, {}, clear=True), patch(
            "src.ai_summary.OpenAI"
        ) as openai_client:
            result = generate_ai_summary_openai(data)

        openai_client.assert_not_called()
        self.assertIn("OPENAI_API_KEY", result)


if __name__ == "__main__":
    unittest.main()
