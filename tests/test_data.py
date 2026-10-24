import tempfile
import unittest
from pathlib import Path

from finetune.data import grouped_split, load_preference_rows, load_sft_rows


class DataTests(unittest.TestCase):
    def setUp(self):
        self.target = {
            "vendor": "Northwind",
            "invoice_number": "INV-1",
            "date": None,
            "total": 12.5,
            "currency": "USD",
        }

    def test_sft_loader_formats_prompt_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            path.write_text(
                '{"text":"Invoice 1","target":{"vendor":"Northwind","invoice_number":"INV-1","date":null,"total":12.5,"currency":"USD"}}\n',
                encoding="utf-8",
            )
            rows = load_sft_rows(path)
        self.assertEqual(len(rows), 1)
        self.assertIn("Invoice 1", rows[0]["text"])
        self.assertIn('"currency": "USD"', rows[0]["text"])

    def test_sft_loader_accepts_prepared_conversation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.jsonl"
            path.write_text(
                '{"source_id":"source-1","messages":[{"role":"user","content":"Return JSON for Ada."}],"target":{"name":"Ada"}}\n',
                encoding="utf-8",
            )
            rows = load_sft_rows(path)
        self.assertEqual(len(rows), 1)
        self.assertIn("user: Return JSON for Ada.", rows[0]["text"])
        self.assertIn('"name": "Ada"', rows[0]["text"])

    def test_preference_loader_accepts_generic_json_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.jsonl"
            path.write_text(
                '{"source_id":"source-1","messages":[{"role":"user","content":"Return JSON for Ada."}],"chosen":{"name":"Ada"},"rejected":{"name":"Grace"}}\n',
                encoding="utf-8",
            )
            rows = load_preference_rows(path)
        self.assertEqual(len(rows), 1)
        self.assertIn("user: Return JSON for Ada.", rows[0]["prompt"])
        self.assertEqual(rows[0]["chosen"], '{"name": "Ada"}')

    def test_preference_loader_accepts_plain_text_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.jsonl"
            path.write_text(
                '{"source_id":"source-1","messages":[{"role":"user","content":"Describe Ada."}],"chosen":"Ada Lovelace was a mathematician.","rejected":"Ada was a person."}\n',
                encoding="utf-8",
            )
            rows = load_preference_rows(path)
        self.assertEqual(rows[0]["chosen"], "Ada Lovelace was a mathematician.")
        self.assertEqual(rows[0]["rejected"], "Ada was a person.")

    def test_preference_loader_requires_distinct_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.jsonl"
            path.write_text(
                '{"text":"Invoice 1","chosen":{"vendor":"Northwind","invoice_number":"INV-1","date":null,"total":12.5,"currency":"USD"},"rejected":{"vendor":"Northwind","invoice_number":"INV-1","date":null,"total":12.5,"currency":"USD"}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must differ"):
                load_preference_rows(path)

    def test_grouped_split_keeps_sources_disjoint(self):
        rows = [
            {"source_id": f"source-{index}", "text": "text", "target": self.target}
            for index in range(20)
        ]
        splits = grouped_split(rows)
        source_sets = {name: {row["source_id"] for row in split} for name, split in splits.items()}
        self.assertFalse(source_sets["train"] & source_sets["validation"])
        self.assertFalse(source_sets["train"] & source_sets["test"])
        self.assertFalse(source_sets["validation"] & source_sets["test"])
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_training_and_preference_template_prompts_match(self):
        class Tokenizer:
            chat_template = "test"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                rendered = "".join(f"<{item['role']}>{item['content']}" for item in messages)
                return rendered + ("<assistant>" if add_generation_prompt else "")

        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            sft_path = Path(directory) / "sft.jsonl"
            preference_path = Path(directory) / "preferences.jsonl"
            sft_path.write_text(
                '{"text":"Invoice 1","target":{"vendor":"Northwind","invoice_number":"INV-1","date":null,"total":12.5,"currency":"USD"}}\n',
                encoding="utf-8",
            )
            preference_path.write_text(
                '{"text":"Invoice 1","chosen":{"vendor":"Northwind","invoice_number":"INV-1","date":null,"total":12.5,"currency":"USD"},"rejected":{"vendor":"Elsewhere","invoice_number":"INV-1","date":null,"total":12.5,"currency":"USD"}}\n',
                encoding="utf-8",
            )
            sft_text = load_sft_rows(sft_path, tokenizer)[0]["text"]
            preference_prompt = load_preference_rows(preference_path, tokenizer)[0]["prompt"]
        self.assertTrue(sft_text.startswith(preference_prompt))


if __name__ == "__main__":
    unittest.main()
