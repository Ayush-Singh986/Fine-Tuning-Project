import json
import unittest

from finetune.contracts import INVOICE_SCHEMA, parse_json_object
from finetune.metrics import score_predictions


class ContractsTests(unittest.TestCase):
    def test_accepts_valid_object_and_json_fence(self):
        output = {
            "vendor": "Northwind",
            "invoice_number": "INV-1",
            "date": "2025-02-14",
            "total": 12.5,
            "currency": "USD",
        }
        self.assertEqual(parse_json_object(json.dumps(output)), output)
        self.assertEqual(parse_json_object(f"```json\n{json.dumps(output)}\n```"), output)

    def test_rejects_missing_fields_and_wrong_types(self):
        with self.assertRaisesRegex(ValueError, "does not match the supplied schema"):
            parse_json_object('{"vendor":"Northwind"}', INVOICE_SCHEMA)
        invalid = {
            "vendor": "Northwind",
            "invoice_number": "INV-1",
            "date": None,
            "total": True,
            "currency": "USD",
        }
        with self.assertRaisesRegex(ValueError, "does not match the supplied schema"):
            parse_json_object(json.dumps(invalid), INVOICE_SCHEMA)

    def test_rejects_impossible_calendar_date(self):
        with self.assertRaisesRegex(ValueError, "does not match the supplied schema"):
            parse_json_object(
                '{"vendor":"Northwind","invoice_number":"INV-1",'
                '"date":"2025-02-30","total":12.5,"currency":"USD"}',
                INVOICE_SCHEMA,
            )

    def test_metrics_count_invalid_outputs_as_failures(self):
        target = {
            "vendor": "Northwind",
            "invoice_number": "INV-1",
            "date": None,
            "total": 12.5,
            "currency": "USD",
        }
        report = score_predictions([target, target], [json.dumps(target), "not json"])
        self.assertEqual(report["examples"], 2)
        self.assertEqual(report["json_parse_rate"], 0.5)
        self.assertEqual(report["exact_match_accuracy"], 0.5)
        self.assertEqual(report["field_value_accuracy"], 0.5)

    def test_metrics_distinguish_parse_success_from_schema_validity(self):
        target = {
            "vendor": "Northwind",
            "invoice_number": "INV-1",
            "date": None,
            "total": 12.5,
            "currency": "USD",
        }
        report = score_predictions([target], ['{"vendor":"Northwind"}'])
        self.assertEqual(report["json_parse_rate"], 1.0)
        self.assertEqual(report["exact_match_accuracy"], 0.0)

    def test_schema_metric_is_only_reported_for_examples_with_schema(self):
        target = {"name": "Ada"}
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        report = score_predictions([target], ['{"name":"Ada"}'], [schema])
        self.assertEqual(report["schema_valid_rate"], 1.0)
        self.assertEqual(report["schema_validation_examples"], 1)

    def test_invalid_json_cannot_count_as_exact_match_for_empty_target(self):
        report = score_predictions([{}], ["not json"])
        self.assertEqual(report["json_parse_rate"], 0.0)
        self.assertEqual(report["exact_match_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
