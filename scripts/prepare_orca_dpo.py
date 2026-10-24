import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from finetune.data import grouped_split


def prepare_rows(source: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in parquet.ParquetFile(source).iter_batches(
        batch_size=512,
        columns=["system", "input", "chosen", "rejected"],
    ):
        for row in batch.to_pylist():
            counts["source_rows"] += 1
            user_text = row.get("input")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            if not isinstance(user_text, str) or not user_text.strip():
                counts["missing_prompt"] += 1
                continue
            if not isinstance(chosen, str) or not chosen.strip():
                counts["missing_chosen"] += 1
                continue
            if not isinstance(rejected, str) or not rejected.strip():
                counts["missing_rejected"] += 1
                continue
            if chosen.strip() == rejected.strip():
                counts["identical_responses"] += 1
                continue
            system_text = row.get("system")
            messages = []
            if isinstance(system_text, str) and system_text.strip():
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": user_text})

            identity = hashlib.sha256(
                json.dumps(
                    {
                        "messages": messages,
                        "chosen": chosen.strip(),
                        "rejected": rejected.strip(),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if identity in seen:
                counts["duplicates_removed"] += 1
                continue
            seen.add(identity)
            prompt_id = hashlib.sha256(
                json.dumps(
                    messages,
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "source_id": f"orca:{prompt_id}",
                    "source_dataset": "argilla/distilabel-intel-orca-dpo-pairs",
                    "messages": messages,
                    "chosen": chosen.strip(),
                    "rejected": rejected.strip(),
                }
            )
            counts["accepted"] += 1
    return rows, dict(counts)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and split the downloaded Argilla Orca preference pairs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/orca_dpo_pairs/data/train-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared/orca-dpo"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    rows, counts = prepare_rows(args.input)
    if not rows:
        raise ValueError("No valid preference pairs were found.")
    splits = grouped_split(
        rows,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    write_jsonl(args.output_dir / "all.jsonl", rows)
    for name, split in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
    report = {
        "source_dataset": "argilla/distilabel-intel-orca-dpo-pairs",
        "source_revision": "0b10ec0df32c919f95126b203c8f5962b6875896",
        "source_license": "Apache-2.0",
        "filter": (
            "Non-empty prompt, chosen, and rejected strings; identical responses "
            "and exact duplicate pairs removed."
        ),
        "source_counts": counts,
        "prepared_pairs": len(rows),
        "split_counts": {name: len(split) for name, split in splits.items()},
        "human_reviewed_for_json_task": False,
        "warning": (
            "These are general chat preference pairs, not JSON-task-specific "
            "preferences. Evaluate for JSON-format regressions or replace with "
            "human-reviewed task-specific pairs."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
