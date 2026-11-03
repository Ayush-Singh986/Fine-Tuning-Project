# Import the Any type hint for flexible type annotations
from typing import Any

# Import JSON parsing and validation functions from our contracts module
# decode_json_object: Parses JSON from model output (handles markdown)
# validate_json_object: Validates against a JSON Schema
from finetune.contracts import decode_json_object, validate_json_object


def score_predictions(
    expected: list[dict[str, Any]],
    generated: list[str],
    schemas: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Evaluate model predictions against ground truth targets.
    
    This function computes comprehensive metrics for structured JSON extraction:
    - JSON parse rate: Can the model output be parsed as valid JSON?
    - Exact match accuracy: Does the prediction match the target exactly?
    - Field-level accuracy: How well does each field perform individually?
    - Schema validation rate: Does the output conform to the schema?
    
    These metrics help identify specific failure modes:
    - Low parse rate → Model is outputting invalid JSON format
    - Low exact match → Model is extracting values but not exactly right
    - Field-level accuracy → Which specific fields are problematic
    
    Args:
        expected: List of ground truth JSON objects (the correct answers)
        generated: List of model-generated JSON strings (raw output)
        schemas: Optional list of JSON Schemas (one per example, or None)
        
    Returns:
        A dictionary containing all evaluation metrics
        
    Raises:
        ValueError: If inputs are mismatched or empty
    """
    # Ensure we have the same number of expected and generated outputs
    # This prevents silent errors where predictions don't match targets
    if len(expected) != len(generated):
        raise ValueError("Expected target and generated output counts must match.")
    # If schemas are provided, they must match the example count
    if schemas is not None and len(schemas) != len(expected):
        raise ValueError("Schema and target counts must match.")
    # Cannot compute meaningful metrics on an empty test set
    if not expected:
        raise ValueError("Cannot evaluate an empty test set.")

    # Initialize counters for aggregate metrics
    parse_success = 0  # Number of outputs that successfully parsed as JSON
    exact_match = 0  # Number of outputs that exactly match the target
    schema_valid = 0  # Number of outputs that pass schema validation
    schema_examples = 0  # Number of examples with a schema provided
    
    # Track field-level accuracy to identify which fields are problematic
    # field_correct[field] = number of correct predictions for this field
    field_correct: dict[str, int] = {}
    # field_total[field] = total number of examples with this field
    field_total: dict[str, int] = {}
    
    # Iterate over each example and its corresponding model output
    for index, (target, output) in enumerate(zip(expected, generated)):
        # Get the schema for this example (if one was provided)
        schema = schemas[index] if schemas is not None else None
        # Validate the target against the schema (raises if invalid)
        # This ensures our ground truth is correct before evaluating
        target = validate_json_object(target, schema)
        
        # Try to parse the model's output as JSON
        parsed = True  # Assume parsing will succeed
        try:
            prediction = decode_json_object(output)
        except (ValueError, TypeError):
            # If parsing fails, use an empty dict as the prediction
            # This allows us to still compute metrics (like parse rate)
            prediction = {}
            parsed = False
        else:
            # Parsing succeeded, increment the success counter
            parse_success += 1
        
        # If a schema was provided, check if the prediction validates
        if schema is not None:
            schema_examples += 1
            if parsed:
                try:
                    validate_json_object(prediction, schema)
                except ValueError:
                    # Schema validation failed, don't increment counter
                    pass
                else:
                    # Schema validation succeeded
                    schema_valid += 1
        
        # Check for exact match (only if parsing succeeded)
        # This is a strict metric: all fields must match exactly
        exact_match += parsed and prediction == target
        
        # Compute field-level accuracy
        # We check all fields that appear in either the target or prediction
        for field in target.keys() | prediction.keys():
            # Count how many examples had this field
            field_total[field] = field_total.get(field, 0) + 1
            # Initialize the correct counter for this field if needed
            field_correct.setdefault(field, 0)
            # Check if this field was predicted correctly
            # Both the field must exist in both dicts AND have the same value
            if field in target and field in prediction and prediction[field] == target[field]:
                field_correct[field] = field_correct.get(field, 0) + 1

    # Compute final metrics from the counters
    count = len(expected)
    
    # Compute per-field accuracy
    # This helps identify which specific fields are problematic
    field_accuracy = {
        field: correct / field_total[field]
        for field, correct in sorted(field_correct.items())
    }
    
    # Compute overall field-value accuracy across all fields
    # This is a single number summarizing field-level performance
    field_value_accuracy = sum(field_correct.values()) / max(1, sum(field_total.values()))
    
    # Build the final report dictionary
    report = {
        # Total number of examples evaluated
        "examples": count,
        # Percentage of outputs that successfully parsed as JSON
        "json_parse_rate": parse_success / count,
        # Percentage of outputs that exactly match the target
        "exact_match_accuracy": exact_match / count,
        # Per-field accuracy breakdown
        "field_accuracy": field_accuracy,
        # Overall field-value accuracy
        "field_value_accuracy": field_value_accuracy,
    }
    
    # If schema validation was performed, include those metrics
    if schema_examples:
        report["schema_validation_examples"] = schema_examples
        report["schema_valid_rate"] = schema_valid / schema_examples
    
    return report
