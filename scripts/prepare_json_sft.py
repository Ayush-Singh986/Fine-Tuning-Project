import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from finetune.data import grouped_split


def parse_final_json(messages: Any) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    if not isinstance(messages, list) or not messages:
        return None
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            continue
        normalized.append({"role": role, "content": content})
    if not normalized or normalized[-1]["role"] != "assistant":
        return None
    try:
        target = json.loads(normalized[-1]["content"])
    except json.JSONDecodeError:
        return None
    if not isinstance(target, dict):
        return None
    prompt_messages = normalized[:-1]
    if not prompt_messages or not any(message["role"] == "user" for message in prompt_messages):
        return None
    return prompt_messages, target


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid source JSONL: {error.msg}"
                ) from error


def load_candidates(
    structured_outputs: Path,
    chat_if: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    sources = (
        ("nemotron_structured_outputs", structured_outputs, "any"),
        ("nemotron_chat_if", chat_if, "last_user_json"),
    )
    for source_name, source_path, filter_type in sources:
        for _, row in iter_jsonl(source_path):
            if not isinstance(row, dict):
                counts[f"{source_name}_non_object_rows"] += 1
                continue
            counts[f"{source_name}_rows"] += 1
            result = parse_final_json(row.get("messages"))
            if result is None:
                continue
            prompt_messages, target = result
            if filter_type == "any":
                prompt = "\n".join(
                    message["content"]
                    for message in prompt_messages
                    if message["role"] in {"system", "user"}
                )
                if "json" not in prompt.lower():
                    continue
            else:
                last_user = next(
                    (
                        message["content"]
                        for message in reversed(prompt_messages)
                        if message["role"] == "user"
                    ),
                    "",
                )
                if "json" not in last_user.lower():
                    continue

            source_id = f"{source_name}:{row.get('uuid') or counts[source_name + '_rows']}"
            identity = hashlib.sha256(
                json.dumps(
                    {"messages": prompt_messages, "target": target},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if identity in seen:
                counts["duplicates_removed"] += 1
                continue
            seen.add(identity)
            candidates.append(
                {
                    "source_id": source_id,
                    "source_dataset": source_name,
                    "messages": prompt_messages,
                    "target": target,
                }
            )
            counts[f"{source_name}_accepted"] += 1

    return candidates, dict(counts)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter valid JSON-output examples from the downloaded Nemotron corpus."
    )
    parser.add_argument(
        "--structured-outputs",
        type=Path,
        default=Path("data/external/nemotron_instruction_chat/data/structured_outputs.jsonl"),
    )
    parser.add_argument(
        "--chat-if",
        type=Path,
        default=Path("data/external/nemotron_instruction_chat/data/chat_if.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared/structured-json"),
    )
    parser.add_argument("--min-examples", type=int, default=2000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    rows, counts = load_candidates(args.structured_outputs, args.chat_if)
    print(json.dumps(counts, indent=2))
    if len(rows) < args.min_examples:
        raise ValueError(
            f"Only {len(rows)} valid JSON candidates passed the filters; "
            f"the requested minimum is {args.min_examples}. No prepared split was written."
        )

    splits = grouped_split(
        rows,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    write_jsonl(args.output_dir / "all.jsonl", rows)
    for name, split in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
        print(f"{name}: {len(split)} examples")

    report = {
        "source_revisions": {
            "nvidia/Nemotron-Instruction-Following-Chat-v1": "83dcd3aded0d289b0bbc018d3f9af4c5dd4005df"
        },
        "filter": (
            "Final assistant message must be a strict JSON object. "
            "For structured_outputs, system/user instructions must mention JSON; "
            "for chat_if, the final user message must mention JSON."
        ),
        "source_counts": counts,
        "prepared_examples": len(rows),
        "split_counts": {name: len(split) for name, split in splits.items()},
        "human_reviewed": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared data and report written to {args.output_dir}")


if __name__ == "__main__":
    main()
