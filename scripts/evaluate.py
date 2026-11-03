# Import argparse for command-line argument parsing
import argparse
# Import hashlib for computing checksums of test sets
import hashlib
# Import json for reading and writing JSON files
import json
# Import os for platform-specific line endings
import os
# Import Path for cross-platform file path handling
from pathlib import Path
# Import Any for flexible type annotations
from typing import Any

# Import our data loading function
from finetune.data import read_jsonl
# Import our evaluation metrics
from finetune.metrics import score_predictions
# Import our model inference class
from finetune.model import TransformersExtractor


def main() -> None:
    """Evaluate a trained model on a held-out test set.
    
    This script runs the full evaluation pipeline:
    1. Load the test examples
    2. Generate predictions from the model
    3. Compute comprehensive metrics
    4. Save a detailed evaluation report
    
    The report includes:
    - JSON parse rate: Can the model output valid JSON?
    - Exact match accuracy: Does the output match the target exactly?
    - Field-level accuracy: How well does each field perform?
    - Schema validation rate: Does output conform to the schema?
    
    Usage:
        python scripts/evaluate.py --test data/test.jsonl \\
            --model-path output/sft/adapter --output reports/sft.json
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Evaluate a model on held-out structured JSON examples.")
    # Path to the test set (JSONL format with messages/text and target)
    parser.add_argument("--test", type=Path, required=True, help="JSONL with messages and target, or text and target.")
    # Path to the model (either merged model or LoRA adapter)
    parser.add_argument("--model-path", required=True, help="Merged model directory or LoRA adapter directory.")
    # Optional override for the base model (useful if adapter config is incorrect)
    parser.add_argument("--base-model", help="Override the base model recorded in an adapter.")
    # Optional path to save the evaluation report
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    # Load the test examples
    rows = read_jsonl(args.test)
    
    # Initialize the model for inference
    # This loads the model and prepares it for generation
    extractor = TransformersExtractor(args.model_path, args.base_model)
    
    # Generate predictions for all test examples
    # We use a list comprehension for clarity
    predictions = [
        # Check if the row uses the chat format (messages array)
        extractor.generate(row["messages"])
        if isinstance(row.get("messages"), list)
        # Otherwise, use the simple format (wrap text in a user message)
        else extractor.generate([{"role": "user", "content": row["text"]}])
        for row in rows
    ]
    
    # Compute evaluation metrics
    # This compares predictions to ground truth targets
    report: dict[str, Any] = score_predictions(
        # Ground truth targets
        [row["target"] for row in rows],
        # Model predictions
        predictions,
        # Optional schemas for validation
        [row.get("schema") for row in rows]
        if any(row.get("schema") is not None for row in rows)
        else None,
    )
    
    # Add metadata about the evaluation to the report
    # Model path for reproducibility
    report["model_path"] = args.model_path
    # Test file path for reference
    report["test_file"] = str(args.test)
    # SHA256 hash of the test file to detect if the test set changes
    # This ensures we're comparing results from the same test set
    report["test_set_sha256"] = hashlib.sha256(args.test.read_bytes()).hexdigest()
    # Generation parameters for reproducibility
    report["generation"] = {"do_sample": False, "max_new_tokens": 512}
    
    # Format the report as pretty-printed JSON
    encoded = json.dumps(report, indent=2)
    # Print to console for immediate feedback
    print(encoded)
    
    # Save the report to a file if requested
    if args.output:
        # Create parent directories if they don't exist
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Write the report with platform-specific line endings
        args.output.write_text(encoded + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
