# Import argparse for command-line argument parsing
import argparse
# Import json for reading JSON report files
import json
# Import Path for cross-platform file path handling
from pathlib import Path
# Import Any for flexible type annotations
from typing import Any

# Define the core metrics to compare across training stages
# Each tuple contains (display_name, json_key)
# These are the key metrics that demonstrate improvement from fine-tuning
METRICS = (
    # JSON parse rate: Can the model output valid JSON?
    ("JSON parse rate", "json_parse_rate"),
    # Exact-match accuracy: Does the output match the target exactly?
    ("Exact-match accuracy", "exact_match_accuracy"),
    # Field-value accuracy: How many field values are correct?
    ("Field-value accuracy", "field_value_accuracy"),
)


def load_report(path: Path) -> dict[str, Any]:
    """Load and validate an evaluation report.
    
    This function loads a JSON report file and validates that it contains
    all required metrics. This ensures we can make fair comparisons across
    different training stages.
    
    Args:
        path: Path to the JSON report file
        
    Returns:
        The loaded report as a dictionary
        
    Raises:
        ValueError: If required metrics are missing
    """
    # Open and load the JSON report
    with path.open(encoding="utf-8") as source:
        report = json.load(source)
    
    # Validate that all required metrics are present and numeric
    for label, key in METRICS:
        if not isinstance(report.get(key), (int, float)):
            raise ValueError(f"{path}: missing numeric metric {key!r} ({label}).")
    
    return report


def main() -> None:
    """Compare evaluation reports across training stages.
    
    This script compares three evaluation reports:
    1. Base prompted model (no fine-tuning)
    2. After SFT (Supervised Fine-Tuning)
    3. After DPO (Direct Preference Optimization)
    
    It generates a Markdown table showing the progression of metrics,
    making it easy to demonstrate the value of each training stage.
    
    Usage:
        python scripts/compare_reports.py \\
            --base reports/base.json \\
            --sft reports/sft.json \\
            --dpo reports/dpo.json \\
            --output reports/comparison.md
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare base, SFT, and DPO evaluation reports.")
    # Path to the base model evaluation report
    parser.add_argument("--base", type=Path, required=True)
    # Path to the SFT model evaluation report
    parser.add_argument("--sft", type=Path, required=True)
    # Path to the DPO model evaluation report
    parser.add_argument("--dpo", type=Path, required=True)
    # Optional output path for the Markdown comparison table
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    # Load all three reports into a dictionary for easy comparison
    reports = {
        "Base prompted": load_report(args.base),
        "After SFT": load_report(args.sft),
        "After DPO": load_report(args.dpo),
    }
    
    # Validate that all reports used the same number of test examples
    # This ensures a fair comparison
    counts = {report.get("examples") for report in reports.values()}
    if len(counts) != 1:
        raise ValueError("All reports must use the same number of held-out examples.")
    
    # Validate that all reports used the same test set
    # We check the SHA256 hash of the test file
    test_hashes = {report.get("test_set_sha256") for report in reports.values()}
    if len(test_hashes) != 1 or None in test_hashes:
        raise ValueError("All reports must use the same held-out test-set contents.")
    
    # Validate that all reports used the same generation settings
    # Different settings (e.g., temperature) could affect results
    generations = {
        json.dumps(report.get("generation"), sort_keys=True)
        for report in reports.values()
    }
    if len(generations) != 1 or "null" in generations:
        raise ValueError("All reports must use the same decoding settings.")

    # Build the Markdown comparison table
    # Header row with column names
    lines = [
        "| Metric | Base prompted | After SFT | After DPO |",
        # Separator row with right-aligned number columns
        "|---|---:|---:|---:|",
    ]
    
    # Add a row for each metric
    for label, key in METRICS:
        # Format each value as a percentage with 2 decimal places
        values = [f"{reports[stage][key] * 100:.2f}%" for stage in reports]
        # Build the table row
        lines.append(f"| {label} | {' | '.join(values)} |")
    
    # Check if schema validation was performed
    schema_reports = [
        report.get("schema_valid_rate")
        for report in reports.values()
    ]
    
    # Add schema validation row if available
    if all(isinstance(value, (int, float)) for value in schema_reports):
        # All reports have schema validation - include it
        values = [f"{value * 100:.2f}%" for value in schema_reports]
        lines.append(f"| Schema-valid rate | {' | '.join(values)} |")
    elif any(value is not None for value in schema_reports):
        # Some reports have schema validation but not all - this is an error
        raise ValueError("Schema-valid rate must be available in all reports or none.")
    else:
        # No reports have schema validation - show N/A
        lines.append("| Schema-valid rate | N/A | N/A | N/A |")
    
    # Add a note about the test set size
    lines.append(f"\nHeld-out examples: {next(iter(counts))}")
    
    # Join all lines into the final Markdown text
    result = "\n".join(lines)
    
    # Print to console for immediate feedback
    print(result)
    
    # Write to file if requested
    if args.output:
        # Create parent directories if they don't exist
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Write the Markdown content
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
