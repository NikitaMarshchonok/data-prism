import unittest

from vibedash.generator_bridge import _format_value
from vibedash.spec import VizSpec, create_saas_demo_viz_spec


class VibeDashDemoSpecTests(unittest.TestCase):
    def test_demo_spec_is_stable_and_targets_demo_columns(self):
        first = create_saas_demo_viz_spec()
        second = create_saas_demo_viz_spec()

        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(len(first.metrics), 4)
        self.assertEqual(len(first.charts), 5)
        self.assertEqual(first.charts[0].x, "Date")
        self.assertEqual(first.charts[0].y, "MonthlyRevenue")

    def test_viz_spec_collection_defaults_are_not_shared(self):
        first = VizSpec(title="First")
        second = VizSpec(title="Second")

        first.comments.append("Only on the first")

        self.assertEqual(second.comments, [])

    def test_ratio_percentages_are_rendered_as_human_percentages(self):
        self.assertEqual(_format_value(0.067, "percent"), "6.7%")
        self.assertEqual(_format_value(6.7, "percent"), "6.7%")


if __name__ == "__main__":
    unittest.main()
