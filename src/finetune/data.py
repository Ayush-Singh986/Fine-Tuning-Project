import json
from pathlib import Path
from typing import Any

from finetune.contracts import (
    conversation_messages,
    make_prompt,
    validate_json_object,
    validate_target,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object.")
            rows.append(row)
    return rows


def load_sft_rows(path: Path, tokenizer: Any | None = None) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    prepared: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row.get("messages"), list):
            target = validate_json_object(row.get("target"))
            messages = row["messages"]
            if not messages or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"{path}: row {index}: messages must contain valid role/content objects.")
            messages = messages + [
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
            ]
            serialized = (
                tokenizer.apply_chat_template(messages, tokenize=False)
                if tokenizer is not None and tokenizer.chat_template
                else "\n".join(
                    f"{message['role']}: {message['content']}" for message in messages
                )
            )
            prepared.append({"text": serialized})
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: row {index}: text must be a non-empty string.")
        target = validate_target(row.get("target"))
        answer = json.dumps(target, ensure_ascii=False)
        messages = conversation_messages(text) + [{"role": "assistant", "content": answer}]
        serialized = (
            tokenizer.apply_chat_template(messages, tokenize=False)
            if tokenizer is not None and tokenizer.chat_template
            else f"{make_prompt(text)}\n{answer}"
        )
        prepared.append({"text": serialized})
    if not prepared:
        raise ValueError(f"{path}: dataset is empty.")
    return prepared


def load_preference_rows(path: Path, tokenizer: Any | None = None) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    prepared: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row.get("messages"), list):
            messages = row["messages"]
            if not messages or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"{path}: row {index}: messages must contain valid role/content objects.")
            chosen = _preference_response(row.get("chosen"), path, index, "chosen")
            rejected = _preference_response(row.get("rejected"), path, index, "rejected")
            if chosen == rejected:
                raise ValueError(f"{path}: row {index}: chosen and rejected must differ.")
            prompt = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if tokenizer is not None and tokenizer.chat_template
                else "\n".join(
                    f"{message['role']}: {message['content']}"
                    for message in messages
                )
            )
            prepared.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: row {index}: text must be a non-empty string.")
        chosen = _preference_response(row.get("chosen"), path, index, "chosen")
        rejected = _preference_response(row.get("rejected"), path, index, "rejected")
        if chosen == rejected:
            raise ValueError(f"{path}: row {index}: chosen and rejected must differ.")
        messages = conversation_messages(text)
        prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if tokenizer is not None and tokenizer.chat_template
            else make_prompt(text)
        )
        prepared.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    if not prepared:
        raise ValueError(f"{path}: dataset is empty.")
    return prepared


def _preference_response(
    value: Any,
    path: Path,
    row_index: int,
    field: str,
) -> str:
    if isinstance(value, dict):
        return json.dumps(validate_json_object(value), ensure_ascii=False)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(
        f"{path}: row {row_index}: {field} must be a non-empty string or JSON object."
    )


def grouped_split(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Split by source_id so pages from the same document cannot leak."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation and test fractions must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1.")

    groups: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows, start=1):
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"row {index}: source_id must be a non-empty string.")
        groups.setdefault(source_id, []).append(row)
    if len(groups) < 3:
        raise ValueError("At least three distinct source_id values are required.")

    # Stable ordering makes the split reproducible without relying on random state.
    import hashlib

    source_ids = sorted(
        groups,
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    test_count = max(1, round(len(source_ids) * test_fraction))
    validation_count = max(1, round(len(source_ids) * validation_fraction))
    if test_count + validation_count >= len(source_ids):
        raise ValueError("Not enough source groups for the requested split fractions.")
    test_ids = set(source_ids[:test_count])
    validation_ids = set(source_ids[test_count : test_count + validation_count])
    return {
        "train": [row for row in rows if row["source_id"] not in test_ids | validation_ids],
        "validation": [row for row in rows if row["source_id"] in validation_ids],
        "test": [row for row in rows if row["source_id"] in test_ids],
    }
