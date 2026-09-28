import json
import unittest

from vibedash.outcome_measurement import (
    build_outcome_measurement,
    deserialize_outcome_measurement,
    normalize_outcome_measurement,
    serialize_outcome_measurement,
)


class OutcomeMeasurementTests(unittest.TestCase):
    def test_complete_measurement_is_canonical_and_calculates_change(self):
        measurement = build_outcome_measurement(
            baseline_value="42.000",
            observed_value="48",
            unit=" percentage points ",
            period_start="2026-09-01",
            period_end="2026-09-30",
            required=True,
        )

        self.assertEqual(measurement["baseline_value"], "42")
        self.assertEqual(measurement["observed_value"], "48")
        self.assertEqual(measurement["delta_value"], "6")
        self.assertEqual(measurement["relative_change_percent"], 14.29)
        self.assertEqual(measurement["unit"], "percentage points")

    def test_zero_baseline_has_no_invented_percentage_change(self):
        measurement = build_outcome_measurement(
            baseline_value="0",
            observed_value="5",
            unit="customers",
            period_start="2026-09-01",
            period_end="2026-09-30",
        )

        self.assertEqual(measurement["delta_value"], "5")
        self.assertIsNone(measurement["relative_change_percent"])

    def test_optional_measurement_is_all_or_nothing(self):
        self.assertEqual(build_outcome_measurement(), {})
        with self.assertRaisesRegex(ValueError, "require baseline"):
            build_outcome_measurement(required=True)
        with self.assertRaisesRegex(ValueError, "Complete every"):
            build_outcome_measurement(baseline_value="42")

    def test_numbers_dates_and_bounds_fail_closed(self):
        base = {
            "baseline_value": "42",
            "observed_value": "48",
            "unit": "%",
            "period_start": "2026-09-01",
            "period_end": "2026-09-30",
        }
        invalid_updates = (
            {"baseline_value": "NaN"},
            {"baseline_value": "1e3"},
            {"baseline_value": "1.123456789"},
            {"observed_value": "1e19"},
            {"period_start": "2026-10-01"},
            {"unit": "x" * 41},
        )
        for update in invalid_updates:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    build_outcome_measurement(**{**base, **update})

    def test_serialization_omits_derived_values_and_recomputes_them(self):
        measurement = build_outcome_measurement(
            baseline_value="100",
            observed_value="90",
            unit="USD",
            period_start="2026-09-01",
            period_end="2026-09-30",
        )

        serialized = serialize_outcome_measurement(measurement)
        retained = json.loads(serialized)
        restored = deserialize_outcome_measurement(serialized)

        self.assertNotIn("delta_value", retained)
        self.assertNotIn("relative_change_percent", retained)
        self.assertEqual(restored["delta_value"], "-10")
        self.assertEqual(restored["relative_change_percent"], -10.0)

    def test_retained_contract_is_validated(self):
        with self.assertRaisesRegex(ValueError, "contract"):
            normalize_outcome_measurement({"contract": "unknown"})
        with self.assertRaisesRegex(ValueError, "storage"):
            deserialize_outcome_measurement("not-json")


if __name__ == "__main__":
    unittest.main()
