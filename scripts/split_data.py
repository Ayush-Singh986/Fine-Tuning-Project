import argparse
import json
from pathlib import Path

from finetune.data import grouped_split, read_jsonl
from finetune.contracts import validate_json_object, validate_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split task examples by source_id.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows with source_id and target.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, help="Optional preference JSONL sharing source_id values with the examples.")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    for index, row in enumerate(rows, start=1):
        if isinstance(row.get("messages"), list):
            if not row["messages"] or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in row["messages"]
            ):
                raise ValueError(f"row {index}: messages must contain role/content objects.")
            validate_json_object(row.get("target"))
        else:
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"row {index}: text must be a non-empty string.")
            validate_target(row.get("target"))
    splits = grouped_split(
        rows,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    preference_splits = None
    if args.preferences:
        preferences = read_jsonl(args.preferences)
        for index, row in enumerate(preferences, start=1):
            if not isinstance(row.get("source_id"), str) or not row["source_id"].strip():
                raise ValueError(f"preference row {index}: source_id must be a non-empty string.")
            if isinstance(row.get("messages"), list):
                if not row["messages"] or any(
                    not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    for message in row["messages"]
                ):
                    raise ValueError(f"preference row {index}: messages must contain role/content objects.")
                for field in ("chosen", "rejected"):
                    value = row.get(field)
                    if isinstance(value, dict):
                        validate_json_object(value)
                    elif not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"preference row {index}: {field} must be a non-empty string or object."
                        )
                if row["chosen"] == row["rejected"]:
                    raise ValueError(f"preference row {index}: chosen and rejected must differ.")
            else:
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError(f"preference row {index}: text must be a non-empty string.")
                validate_target(row.get("chosen"))
                validate_target(row.get("rejected"))
            if row["chosen"] == row["rejected"]:
                raise ValueError(f"preference row {index}: chosen and rejected must differ.")
        source_split = {
            row["source_id"]: name
            for name, split in splits.items()
            for row in split
        }
        unknown_sources = sorted(
            {row["source_id"] for row in preferences} - source_split.keys()
        )
        if unknown_sources:
            raise ValueError(
                "Preference source_id values must also appear in the SFT input; "
                f"unknown values: {', '.join(unknown_sources[:5])}"
            )
        preference_splits = {name: [] for name in splits}
        for row in preferences:
            preference_splits[source_split[row["source_id"]]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        output = args.output_dir / f"{name}.jsonl"
        with output.open("w", encoding="utf-8") as destination:
            for row in split:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} examples -> {output}")
        if preference_splits is not None:
            preference_output = args.output_dir / f"preferences_{name}.jsonl"
            with preference_output.open("w", encoding="utf-8") as destination:
                for row in preference_splits[name]:
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"preferences {name}: {len(preference_splits[name])} pairs "
                f"-> {preference_output}"
            )


if __name__ == "__main__":
    main()
