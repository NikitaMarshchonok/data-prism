import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VibeDashHistoryUITests(unittest.TestCase):
    def test_navigation_keeps_scope_and_actions_readable(self):
        css = (ROOT / "static" / "vibedash_history.css").read_text(
            encoding="utf-8"
        )

        self.assertIn(".nav-actions .button { width:auto; white-space:nowrap; }", css)
        self.assertIn(".scope-label { flex-shrink:0;", css)
        self.assertIn("white-space:nowrap; }", css)


if __name__ == "__main__":
    unittest.main()
