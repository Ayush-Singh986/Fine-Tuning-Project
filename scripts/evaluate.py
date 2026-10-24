import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from finetune.data import read_jsonl
from finetune.metrics import score_predictions
from finetune.model import TransformersExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on held-out structured JSON examples.")
    parser.add_argument("--test", type=Path, required=True, help="JSONL with messages and target, or text and target.")
    parser.add_argument("--model-path", required=True, help="Merged model directory or LoRA adapter directory.")
    parser.add_argument("--base-model", help="Override the base model recorded in an adapter.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    rows = read_jsonl(args.test)
    extractor = TransformersExtractor(args.model_path, args.base_model)
    predictions = [
        extractor.generate(row["messages"])
        if isinstance(row.get("messages"), list)
        else extractor.generate([{"role": "user", "content": row["text"]}])
        for row in rows
    ]
    report: dict[str, Any] = score_predictions(
        [row["target"] for row in rows],
        predictions,
        [row.get("schema") for row in rows]
        if any(row.get("schema") is not None for row in rows)
        else None,
    )
    report["model_path"] = args.model_path
    report["test_file"] = str(args.test)
    report["test_set_sha256"] = hashlib.sha256(args.test.read_bytes()).hexdigest()
    report["generation"] = {"do_sample": False, "max_new_tokens": 512}
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
