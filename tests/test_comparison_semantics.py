"""Oracles for confirmed meaning, scale agreement, and shared drift roles."""

import copy
import json
import unittest

import numpy as np
import pandas as pd

from src.semantic_contract import (
    SemanticContractError, infer_semantic_contract, parse_semantic_contract_json,
)
from vibedash.comparison_semantics import validate_comparison_semantics
from vibedash.period_comparison import build_period_comparison


class ComparisonSemanticsTests(unittest.TestCase):
    def periods(self):
        before = pd.DataFrame({
            "region": [0, 1] * 20,
            "metric": np.arange(40, dtype=float),
            "conversion_rate": 0.03 + np.arange(40) / 10000,
            "customer_id": [f"PRIVATE-{i}" for i in range(40)],
        })
        after = before.copy()
        after["metric"] += 5
        after["conversion_rate"] += 0.01
        return before, after

    def contracts(self, before, after):
        return tuple(infer_semantic_contract(frame, confirmed=True) for frame in (before, after))

    def test_confirmed_roles_override_both_inference_and_drift_without_rescaling(self):
        before, after = self.periods()
        contracts = self.contracts(before, after)
        for contract in contracts:
            for column in contract["columns"]:
                if column["name"] == "region":
                    column.update(role="measure", unit="number")
                if column["name"] == "metric":
                    column.update(role="category", unit="none", exclude_from_predictors=True)
        report = build_period_comparison(
            before, after,
            baseline_semantic_contract=contracts[0], current_semantic_contract=contracts[1],
        )
        metrics = {metric["column"]: metric for metric in report["numeric_metrics"]["metrics"]}
        drift = {signal["feature"]: signal for signal in report["distribution_drift"]["signals"]}
        self.assertEqual(set(metrics), {"region", "conversion_rate"})
        self.assertEqual(drift["region"]["feature_type"], "numeric")
        self.assertEqual(drift["metric"]["feature_type"], "categorical")
        self.assertAlmostEqual(metrics["conversion_rate"]["absolute_change"], 0.01)
        self.assertEqual(metrics["conversion_rate"]["unit"], "ratio")
        self.assertEqual(report["contract"], "period-comparison-v2")
        self.assertNotIn("PRIVATE-", json.dumps(report, allow_nan=False))
        pd.testing.assert_frame_equal(before, self.periods()[0])

    def test_role_and_scale_mismatch_fail_before_inference(self):
        before, after = self.periods()
        contracts = self.contracts(before, after)
        for field, value in [("unit", "percent"), ("role", "identifier")]:
            changed = copy.deepcopy(contracts[1])
            column = next(item for item in changed["columns"] if item["name"] == "conversion_rate")
            column[field] = value
            if field == "role":
                column.update(unit="none", exclude_from_predictors=True)
            with self.subTest(field=field), self.assertRaisesRegex(SemanticContractError, "different roles or units"):
                build_period_comparison(before, after, baseline_semantic_contract=contracts[0], current_semantic_contract=changed)

    def test_fingerprints_are_canonical_and_added_columns_are_reported(self):
        before, after = self.periods()
        after["new_metric"] = np.arange(40) / 3
        left, right = self.contracts(before, after)
        report = build_period_comparison(before, after, baseline_semantic_contract=left, current_semantic_contract=right)
        repeated = build_period_comparison(before, after, baseline_semantic_contract=left, current_semantic_contract=right)
        self.assertEqual(report, repeated)
        self.assertIn("new_metric", report["schema_changes"]["new_columns"])
        self.assertTrue(report["semantics"]["definitions_aligned"])
        self.assertRegex(report["semantics"]["baseline"]["sha256"], r"^[0-9a-f]{64}$")

    def test_schema_dtype_confirmation_and_unhashable_values_fail_closed(self):
        before, after = self.periods()
        left, right = self.contracts(before, after)
        with self.assertRaises(SemanticContractError):
            validate_comparison_semantics(before, after, left, None)
        for field, value in [("role", []), ("unit", {}), ("confirmed", False)]:
            changed = copy.deepcopy(right)
            if field == "confirmed":
                changed[field] = value
            else:
                changed["columns"][0][field] = value
            with self.subTest(field=field), self.assertRaises(SemanticContractError):
                validate_comparison_semantics(before, after, left, changed)
        with self.assertRaises(SemanticContractError):
            validate_comparison_semantics(before, after.rename(columns={"metric": "changed"}), left, right)
        after["metric"] = "text"
        with self.assertRaisesRegex(SemanticContractError, "not numeric"):
            validate_comparison_semantics(before, after, left, right)
        with self.assertRaises(SemanticContractError):
            parse_semantic_contract_json('{"x":' + '[' * 2000 + '0' + ']' * 2000 + '}')
