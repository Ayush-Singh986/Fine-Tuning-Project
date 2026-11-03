# Import the standard JSON library for parsing JSONL data files
import json
# Import Path for cross-platform file path handling
from pathlib import Path
# Import Any for flexible type annotations
from typing import Any

# Import helper functions from our contracts module
# These functions help with prompt construction and validation
from finetune.contracts import (
    conversation_messages,  # Builds chat-format message lists
    make_prompt,  # Builds single-string prompts for older models
    validate_json_object,  # Validates JSON against a schema
    validate_target,  # Validates training targets against the invoice schema
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries.
    
    JSONL (JSON Lines) format has one JSON object per line.
    This is the standard format for ML training data because:
    - You can stream large files without loading everything into memory
    - Each line is independent, so corrupted lines don't break the whole file
    - Easy to append new examples without rewriting the entire file
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        A list of dictionaries (one per line)
        
    Raises:
        ValueError: If the file contains invalid JSON or non-object rows
    """
    # Initialize the list to store all parsed rows
    rows: list[dict[str, Any]] = []
    # Open the file with UTF-8 encoding (standard for JSON)
    with path.open(encoding="utf-8") as source:
        # Enumerate through lines starting from line number 1 (for error messages)
        for line_number, line in enumerate(source, start=1):
            # Skip empty lines (trailing newlines are common in text files)
            if not line.strip():
                continue
            # Try to parse each line as JSON
            try:
                row = json.loads(line)
            # If JSON parsing fails, raise an error with line number for debugging
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            # Ensure each row is a JSON object (dict), not an array or primitive
            # This is important because our training data expects object rows
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object.")
            # Add the parsed row to our list
            rows.append(row)
    # Return all parsed rows
    return rows


def load_sft_rows(path: Path, tokenizer: Any | None = None) -> list[dict[str, str]]:
    """Load and prepare training examples for supervised fine-tuning (SFT).
    
    This function reads a JSONL file and prepares the text for training.
    It supports two formats:
    1. Chat format: {"messages": [...], "target": {...}}
    2. Simple format: {"text": "...", "target": {...}}
    
    The function serializes the conversation into a single text string that
    the model will learn to generate. If a tokenizer is provided with a
    chat_template, it uses that template for proper formatting.
    
    Args:
        path: Path to the JSONL file containing training examples
        tokenizer: Optional tokenizer with chat_template for formatting
        
    Returns:
        A list of dictionaries with "text" keys containing serialized prompts
        
    Raises:
        ValueError: If the data is invalid or improperly formatted
    """
    # Read the raw JSONL data
    rows = read_jsonl(path)
    # Initialize the list to hold prepared training examples
    prepared: list[dict[str, str]] = []
    
    # Process each row in the dataset
    for index, row in enumerate(rows, start=1):
        # Check if this row uses the chat format (with messages array)
        if isinstance(row.get("messages"), list):
            # Validate and extract the target JSON object
            target = validate_json_object(row.get("target"))
            # Get the messages array
            messages = row["messages"]
            # Validate that messages is not empty and contains valid objects
            # Each message must have a role (system/user/assistant) and string content
            if not messages or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"{path}: row {index}: messages must contain valid role/content objects.")
            # Append the assistant's response (the target) to the messages
            # This creates the full conversation that the model will learn to complete
            messages = messages + [
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
            ]
            # Serialize the conversation into a single text string
            # If the tokenizer has a chat_template, use it for proper formatting
            # Otherwise, use a simple "role: content" format
            serialized = (
                tokenizer.apply_chat_template(messages, tokenize=False)
                if tokenizer is not None and tokenizer.chat_template
                else "\n".join(
                    f"{message['role']}: {message['content']}" for message in messages
                )
            )
            # Add the serialized text to our prepared examples
            prepared.append({"text": serialized})
            # Move to the next row (we've handled this one)
            continue
        
        # Handle the simple format (text + target)
        # Get the input text that describes the extraction task
        text = row.get("text")
        # Validate that text exists and is non-empty
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: row {index}: text must be a non-empty string.")
        # Validate the target against the invoice schema
        # This ensures all training examples have valid targets
        target = validate_target(row.get("target"))
        # Serialize the target to a JSON string for the assistant's response
        answer = json.dumps(target, ensure_ascii=False)
        # Build the conversation in standard chat format
        # conversation_messages creates the system and user messages
        messages = conversation_messages(text) + [{"role": "assistant", "content": answer}]
        # Serialize the conversation (using chat template if available)
        serialized = (
            tokenizer.apply_chat_template(messages, tokenize=False)
            if tokenizer is not None and tokenizer.chat_template
            else f"{make_prompt(text)}\n{answer}"
        )
        # Add the prepared example to our list
        prepared.append({"text": serialized})
    
    # Ensure we have at least some training examples
    if not prepared:
        raise ValueError(f"{path}: dataset is empty.")
    # Return all prepared training examples
    return prepared


def load_preference_rows(path: Path, tokenizer: Any | None = None) -> list[dict[str, str]]:
    """Load and prepare preference pairs for Direct Preference Optimization (DPO).
    
    DPO requires pairs of outputs where one is better than the other:
    - chosen: The preferred output (what we want the model to generate)
    - rejected: The worse output (what we want the model to avoid)
    
    The model learns to maximize the probability of chosen outputs and
    minimize the probability of rejected outputs.
    
    Args:
        path: Path to the JSONL file containing preference pairs
        tokenizer: Optional tokenizer with chat_template for formatting
        
    Returns:
        A list of dictionaries with "prompt", "chosen", and "rejected" keys
        
    Raises:
        ValueError: If preference pairs are invalid or identical
    """
    # Read the raw JSONL data
    rows = read_jsonl(path)
    # Initialize the list to hold prepared preference pairs
    prepared: list[dict[str, str]] = []
    
    # Process each preference pair in the dataset
    for index, row in enumerate(rows, start=1):
        # Check if this row uses the chat format
        if isinstance(row.get("messages"), list):
            # Get the messages array (the prompt conversation)
            messages = row["messages"]
            # Validate that messages contains valid role/content objects
            if not messages or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"{path}: row {index}: messages must contain valid role/content objects.")
            # Extract and validate the chosen (preferred) response
            chosen = _preference_response(row.get("chosen"), path, index, "chosen")
            # Extract and validate the rejected (worse) response
            rejected = _preference_response(row.get("rejected"), path, index, "rejected")
            # Ensure chosen and rejected are different
            # If they're the same, there's nothing for the model to learn
            if chosen == rejected:
                raise ValueError(f"{path}: row {index}: chosen and rejected must differ.")
            # Build the prompt from the messages
            # add_generation_prompt=True adds the assistant prefix
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
            # Add the prepared preference pair
            prepared.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
            # Move to the next row
            continue
        
        # Handle the simple format (text + chosen + rejected)
        # Get and validate the input text
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: row {index}: text must be a non-empty string.")
        # Extract the chosen and rejected responses
        chosen = _preference_response(row.get("chosen"), path, index, "chosen")
        rejected = _preference_response(row.get("rejected"), path, index, "rejected")
        # Ensure they differ (otherwise no learning signal)
        if chosen == rejected:
            raise ValueError(f"{path}: row {index}: chosen and rejected must differ.")
        # Build the prompt messages
        messages = conversation_messages(text)
        # Serialize the prompt
        prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if tokenizer is not None and tokenizer.chat_template
            else make_prompt(text)
        )
        # Add the preference pair
        prepared.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    
    # Ensure we have at least some preference pairs
    if not prepared:
        raise ValueError(f"{path}: dataset is empty.")
    return prepared


def _preference_response(
    value: Any,
    path: Path,
    row_index: int,
    field: str,
) -> str:
    """Extract and validate a preference response (chosen or rejected).
    
    This helper function handles two formats for preference responses:
    1. A JSON object (will be serialized to a string)
    2. A pre-serialized JSON string
    
    Args:
        value: The preference response value (dict or string)
        path: Path to the source file (for error messages)
        row_index: Row number in the file (for error messages)
        field: Field name ("chosen" or "rejected") for error messages
        
    Returns:
        A valid JSON string
        
    Raises:
        ValueError: If the value is not a valid response
    """
    # If the value is a dict, validate it and serialize to JSON
    if isinstance(value, dict):
        return json.dumps(validate_json_object(value), ensure_ascii=False)
    # If the value is a non-empty string, return it (assuming it's already JSON)
    if isinstance(value, str) and value.strip():
        return value.strip()
    # Otherwise, the value is invalid
    raise ValueError(
        f"{path}: row {row_index}: {field} must be a non-empty string or JSON object."
    )


def grouped_split(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Split examples by source_id to prevent data leakage.
    
    When training models, it's critical that examples from the same source
    (e.g., the same document or invoice) don't appear in both training and
    test sets. This prevents the model from memorizing specific documents
    instead of learning general patterns.
    
    For example, if we're extracting from invoices, all pages from the same
    invoice should be in the same split. Otherwise, the model might learn to
    recognize that specific invoice rather than learning to extract from any invoice.
    
    Args:
        rows: List of examples, each with a "source_id" field
        validation_fraction: Fraction of groups to use for validation
        test_fraction: Fraction of groups to use for testing
        
    Returns:
        A dictionary with "train", "validation", and "test" keys
        
    Raises:
        ValueError: If fractions are invalid or there aren't enough groups
    """
    # Validate that fractions are between 0 and 1
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation and test fractions must be between 0 and 1.")
    # Ensure the fractions don't sum to 1 or more (we need training data)
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1.")

    # Group examples by source_id
    # This creates a dict mapping source_id to list of examples from that source
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows, start=1):
        # Extract the source_id from each row
        source_id = row.get("source_id")
        # Validate that source_id exists and is a non-empty string
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"row {index}: source_id must be a non-empty string.")
        # Add this row to the appropriate group
        groups.setdefault(source_id, []).append(row)
    
    # We need at least 3 groups (train, validation, test)
    if len(groups) < 3:
        raise ValueError("At least three distinct source_id values are required.")

    # Sort source_ids using a hash of the id
    # This provides a stable, deterministic ordering without requiring random state
    # The same source_ids will always be in the same split
    import hashlib
    source_ids = sorted(
        groups,
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    
    # Calculate how many groups go to each split
    test_count = max(1, round(len(source_ids) * test_fraction))
    validation_count = max(1, round(len(source_ids) * validation_fraction))
    
    # Ensure we have enough groups for all splits
    if test_count + validation_count >= len(source_ids):
        raise ValueError("Not enough source groups for the requested split fractions.")
    
    # Assign source_ids to each split
    test_ids = set(source_ids[:test_count])
    validation_ids = set(source_ids[test_count : test_count + validation_count])
    
    # Build the final splits by including all examples from each group
    return {
        # Training: all examples from groups not in test or validation
        "train": [row for row in rows if row["source_id"] not in test_ids | validation_ids],
        # Validation: all examples from validation groups
        "validation": [row for row in rows if row["source_id"] in validation_ids],
        # Test: all examples from test groups
        "test": [row for row in rows if row["source_id"] in test_ids],
    }
