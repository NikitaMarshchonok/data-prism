import json
import unittest

import pandas as pd

from src.semantic_contract import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticContractError,
    infer_semantic_contract,
    parse_semantic_contract_json,
    profiles_from_semantic_contract,
    semantic_contract_audit_view,
    validate_semantic_contract,
)


class SemanticContractTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "Date": pd.date_range("2026-01-01", periods=40).astype(str),
            "CustomerID": [f"customer-{index}" for index in range(40)],
            "Revenue": [1000.25 + index * 12.5 for index in range(40)],
            "ConversionRate": [0.02 + index / 10_000 for index in range(40)],
            "Region": ["North", "South"] * 20,
        })

    def _confirmed(self):
        contract = infer_semantic_contract(self.data)
        contract["confirmed"] = True
        return contract

    def test_inference_is_bounded_schema_metadata_with_units_and_safety(self):
        proposal = infer_semantic_contract(self.data)
        columns = {column["name"]: column for column in proposal["columns"]}

        self.assertEqual(proposal["contract"], SEMANTIC_CONTRACT_VERSION)
        self.assertFalse(proposal["confirmed"])
        self.assertEqual(columns["Date"]["role"], "temporal")
        self.assertTrue(columns["Date"]["is_time"])
        self.assertTrue(columns["Date"]["exclude_from_predictors"])
        self.assertEqual(columns["CustomerID"]["role"], "identifier")
        self.assertTrue(columns["CustomerID"]["exclude_from_predictors"])
        self.assertEqual(columns["Revenue"]["unit"], "currency")
        self.assertEqual(columns["ConversionRate"]["unit"], "ratio")
        self.assertNotIn("customer-1", json.dumps(proposal))

    def test_confirmation_produces_stable_fingerprint_and_overridden_profiles(self):
        contract = self._confirmed()
        revenue = next(
            column for column in contract["columns"] if column["name"] == "Revenue"
        )
        revenue["unit"] = "number"
        validated = validate_semantic_contract(self.data, contract)
        repeated = validate_semantic_contract(self.data, contract)
        profiles = profiles_from_semantic_contract(self.data, validated)

        self.assertRegex(validated["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(validated["sha256"], repeated["sha256"])
        self.assertEqual(profiles["Revenue"]["unit"], "number")
        self.assertEqual(profiles["Region"]["role"], "category")
        self.assertEqual(
            semantic_contract_audit_view(validated)["sha256"],
            validated["sha256"],
        )

    def test_unconfirmed_or_schema_mismatched_contract_is_rejected(self):
        proposal = infer_semantic_contract(self.data)
        with self.assertRaisesRegex(SemanticContractError, "confirm"):
            validate_semantic_contract(self.data, proposal)

        proposal["confirmed"] = True
        proposal["columns"][0]["name"] = "Different"
        with self.assertRaisesRegex(SemanticContractError, "no longer matches"):
            validate_semantic_contract(self.data, proposal)

    def test_invalid_role_unit_and_predictive_leakage_are_rejected(self):
        contract = self._confirmed()
        region = next(
            column for column in contract["columns"] if column["name"] == "Region"
        )
        region.update(role="measure", unit="number")
        with self.assertRaisesRegex(SemanticContractError, "not numeric"):
            validate_semantic_contract(self.data, contract)

        contract = self._confirmed()
        customer = next(
            column for column in contract["columns"] if column["name"] == "CustomerID"
        )
        customer["exclude_from_predictors"] = False
        with self.assertRaisesRegex(SemanticContractError, "excluded"):
            validate_semantic_contract(self.data, contract)

        contract = self._confirmed()
        revenue = next(
            column for column in contract["columns"] if column["name"] == "Revenue"
        )
        revenue.update(is_target=True, exclude_from_predictors=False)
        with self.assertRaisesRegex(SemanticContractError, "excluded"):
            validate_semantic_contract(self.data, contract)

    def test_json_parser_rejects_missing_and_non_object_payloads(self):
        with self.assertRaises(SemanticContractError):
            parse_semantic_contract_json("")
        with self.assertRaises(SemanticContractError):
            parse_semantic_contract_json("[]")


if __name__ == "__main__":
    unittest.main()
