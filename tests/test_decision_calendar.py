import unittest

from vibedash.decision_calendar import build_decision_review_calendar


class DecisionCalendarTests(unittest.TestCase):
    def decision_case(self, **overrides):
        value = {
            "id": "a" * 32,
            "created_at": "2026-09-24T10:20:30.000+00:00",
            "review_date": "2026-10-15",
            "owner": "Growth lead",
            "decision": "Test onboarding; compare cohorts, then review.",
            "success_metric": "Activation rate",
            "target_outcome": "Increase activation from 42% to 47%.",
            "evidence_snapshot": {
                "priority": {"title": "Investigate activation decline"},
            },
        }
        value.update(overrides)
        return value

    def test_calendar_is_deterministic_and_escapes_user_text(self):
        calendar = build_decision_review_calendar(self.decision_case()).decode("utf-8")

        self.assertTrue(calendar.startswith("BEGIN:VCALENDAR\r\n"))
        self.assertTrue(calendar.endswith("END:VCALENDAR\r\n"))
        self.assertIn("UID:" + "a" * 32 + "@data-prism.local", calendar)
        self.assertIn("DTSTAMP:20260924T102030Z", calendar)
        self.assertIn("DTSTART;VALUE=DATE:20261015", calendar)
        self.assertIn("DTEND;VALUE=DATE:20261016", calendar)
        self.assertIn("Test onboarding\\; compare cohorts\\, then review.", calendar)
        self.assertNotIn("\nInjected:", calendar.replace("\r\n", ""))
        self.assertEqual(
            calendar,
            build_decision_review_calendar(self.decision_case()).decode("utf-8"),
        )

    def test_calendar_folds_unicode_lines_at_75_octets(self):
        calendar = build_decision_review_calendar(
            self.decision_case(decision="Решение " * 80)
        )

        for line in calendar.split(b"\r\n"):
            self.assertLessEqual(len(line), 75)
        calendar.decode("utf-8")

    def test_calendar_rejects_invalid_contract_fields(self):
        for overrides in (
            {"id": "unsafe"},
            {"review_date": "tomorrow"},
            {"review_date": "9999-12-31"},
            {"created_at": "unknown"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    build_decision_review_calendar(self.decision_case(**overrides))


if __name__ == "__main__":
    unittest.main()
