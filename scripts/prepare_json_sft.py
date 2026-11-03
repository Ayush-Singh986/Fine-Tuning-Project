# Import argparse for command-line argument parsing
import argparse
# Import hashlib for computing deduplication hashes
import hashlib
# Import json for JSON parsing and serialization
import json
# Import Counter for counting occurrences
from collections import Counter
# Import Path for cross-platform file path handling
from pathlib import Path
# Import Any for flexible type annotations
from typing import Any

# Import our data splitting function
from finetune.data import grouped_split


def parse_final_json(messages: Any) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    """Extract a JSON-extraction example from a conversation.
    
    This function examines a conversation history and extracts:
    1. The prompt messages (system + user messages)
    2. The target JSON object (from the final assistant message)
    
    This is used to extract training examples from the Nemotron dataset,
    which contains multi-turn conversations where the assistant outputs JSON.
    
    Args:
        messages: The conversation messages (should be a list of dicts)
        
    Returns:
        Tuple of (prompt_messages, target_json) if valid, None otherwise
    """
    # Validate that messages is a non-empty list
    if not isinstance(messages, list) or not messages:
        return None
    
    # Normalize messages: extract only valid role/content pairs
    normalized: list[dict[str, str]] = []
    for message in messages:
        # Skip non-dict messages
        if not isinstance(message, dict):
            return None
        # Extract role and content
        role = message.get("role")
        content = message.get("content")
        # Only keep valid roles with string content
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            continue
        normalized.append({"role": role, "content": content})
    
    # Ensure the final message is from the assistant (this is our target)
    if not normalized or normalized[-1]["role"] != "assistant":
        return None
    
    # Try to parse the assistant's final message as JSON
    try:
        target = json.loads(normalized[-1]["content"])
    except json.JSONDecodeError:
        return None
    
    # Ensure the target is a JSON object (dict), not an array or primitive
    if not isinstance(target, dict):
        return None
    
    # Extract the prompt messages (everything except the final assistant message)
    prompt_messages = normalized[:-1]
    
    # Ensure there's at least one user message in the prompt
    if not prompt_messages or not any(message["role"] == "user" for message in prompt_messages):
        return None
    
    return prompt_messages, target


def iter_jsonl(path: Path):
    """Iterate over lines in a JSONL file.
    
    This is a generator function that yields (line_number, parsed_json) tuples.
    Using a generator allows processing large files without loading everything into memory.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        Tuples of (line_number, parsed_json_object)
        
    Raises:
        ValueError: If a line contains invalid JSON
    """
    # Open the file with UTF-8 encoding
    with path.open(encoding="utf-8") as source:
        # Enumerate lines starting from 1 for error messages
        for line_number, line in enumerate(source, start=1):
            # Skip empty lines
            if not line.strip():
                continue
            # Try to parse each line as JSON
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
    """Load and filter JSON-extraction examples from the Nemotron dataset.
    
    The Nemotron dataset has two sources:
    1. structured_outputs.jsonl: Examples specifically for structured JSON output
    2. chat_if.jsonl: General instruction-following conversations
    
    We filter for examples where:
    - The final assistant message is a valid JSON object
    - The prompt mentions "json" (to ensure the task is JSON-related)
    
    We also deduplicate examples to prevent training on near-duplicates.
    
    Args:
        structured_outputs: Path to structured_outputs.jsonl
        chat_if: Path to chat_if.jsonl
        
    Returns:
        Tuple of (candidates_list, counts_dict)
    """
    # Counter for tracking statistics
    counts: Counter[str] = Counter()
    # List to hold the filtered candidates
    candidates: list[dict[str, Any]] = []
    # Set to track seen examples (for deduplication)
    seen: set[str] = set()

    # Define the two data sources with their filtering strategies
    sources = (
        # structured_outputs: Check if any system/user message mentions JSON
        ("nemotron_structured_outputs", structured_outputs, "any"),
        # chat_if: Check if the last user message mentions JSON
        ("nemotron_chat_if", chat_if, "last_user_json"),
    )
    
    # Process each source file
    for source_name, source_path, filter_type in sources:
        # Iterate over all examples in the file
        for _, row in iter_jsonl(source_path):
            # Ensure the row is a dictionary
            if not isinstance(row, dict):
                counts[f"{source_name}_non_object_rows"] += 1
                continue
            
            # Count total rows
            counts[f"{source_name}_rows"] += 1
            
            # Try to extract a JSON-extraction example
            result = parse_final_json(row.get("messages"))
            if result is None:
                continue
            
            prompt_messages, target = result
            
            # Apply source-specific filtering
            if filter_type == "any":
                # For structured_outputs, check if any system/user message mentions JSON
                prompt = "\n".join(
                    message["content"]
                    for message in prompt_messages
                    if message["role"] in {"system", "user"}
                )
                if "json" not in prompt.lower():
                    continue
            else:
                # For chat_if, check if the last user message mentions JSON
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

            # Create a unique identifier for this example
            source_id = f"{source_name}:{row.get('uuid') or counts[source_name + '_rows']}"
            
            # Compute a hash of the content for deduplication
            # This ensures we don't include near-duplicate examples
            identity = hashlib.sha256(
                json.dumps(
                    {"messages": prompt_messages, "target": target},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            
            # Skip duplicates
            if identity in seen:
                counts["duplicates_removed"] += 1
                continue
            seen.add(identity)
            
            # Add the candidate to our list
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
    """Write a list of dictionaries to a JSONL file.
    
    JSONL format has one JSON object per line, which is efficient for
    large datasets as you can stream the file without loading it all into memory.
    
    Args:
        path: Output file path
        rows: List of dictionaries to write
    """
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the file for writing with UTF-8 encoding
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            # Write each row as a single line of JSON
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Prepare the training dataset from the Nemotron corpus.
    
    This script:
    1. Loads examples from the Nemotron dataset
    2. Filters for JSON-extraction tasks
    3. Removes duplicates
    4. Splits into train/validation/test sets
    5. Saves the prepared data and a report
    
    Usage:
        python scripts/prepare_json_sft.py \\
            --structured-outputs data/external/nemotron_instruction_chat/data/structured_outputs.jsonl \\
            --chat-if data/external/nemotron_instruction_chat/data/chat_if.jsonl \\
            --output-dir data/prepared/structured-json
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Filter valid JSON-output examples from the downloaded Nemotron corpus."
    )
    # Path to structured_outputs.jsonl
    parser.add_argument(
        "--structured-outputs",
        type=Path,
        default=Path("data/external/nemotron_instruction_chat/data/structured_outputs.jsonl"),
    )
    # Path to chat_if.jsonl
    parser.add_argument(
        "--chat-if",
        type=Path,
        default=Path("data/external/nemotron_instruction_chat/data/chat_if.jsonl"),
    )
    # Output directory for prepared data
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared/structured-json"),
    )
    # Minimum number of examples required
    parser.add_argument("--min-examples", type=int, default=2000)
    # Fraction of data to use for validation
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    # Fraction of data to use for testing
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    # Load and filter candidates from both sources
    rows, counts = load_candidates(args.structured_outputs, args.chat_if)
    # Print statistics
    print(json.dumps(counts, indent=2))
    
    # Ensure we have enough examples
    if len(rows) < args.min_examples:
        raise ValueError(
            f"Only {len(rows)} valid JSON candidates passed the filters; "
            f"the requested minimum is {args.min_examples}. No prepared split was written."
        )

    # Split into train/validation/test
    # grouped_split ensures examples from the same source don't leak across splits
    splits = grouped_split(
        rows,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    
    # Write all examples to a single file
    write_jsonl(args.output_dir / "all.jsonl", rows)
    
    # Write each split to its own file
    for name, split in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
        print(f"{name}: {len(split)} examples")

    # Create a report documenting the data preparation
    report = {
        # Git revision of the source dataset
        "source_revisions": {
            "nvidia/Nemotron-Instruction-Following-Chat-v1": "83dcd3aded0d289b0bbc018d3f9af4c5dd4005df"
        },
        # Description of the filtering criteria
        "filter": (
            "Final assistant message must be a strict JSON object. "
            "For structured_outputs, system/user instructions must mention JSON; "
            "for chat_if, the final user message must mention JSON."
        ),
        # Statistics from the filtering process
        "source_counts": counts,
        # Total number of prepared examples
        "prepared_examples": len(rows),
        # Number of examples in each split
        "split_counts": {name: len(split) for name, split in splits.items()},
        # Flag indicating whether examples were manually reviewed
        # Human review is important for catching errors in the training data
        "human_reviewed": False,
    }
    
    # Write the report
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared data and report written to {args.output_dir}")


if __name__ == "__main__":
    main()
