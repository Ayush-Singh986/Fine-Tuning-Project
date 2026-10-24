from typing import Any

from finetune.contracts import decode_json_object, validate_json_object


def score_predictions(
    expected: list[dict[str, Any]],
    generated: list[str],
    schemas: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    if len(expected) != len(generated):
        raise ValueError("Expected target and generated output counts must match.")
    if schemas is not None and len(schemas) != len(expected):
        raise ValueError("Schema and target counts must match.")
    if not expected:
        raise ValueError("Cannot evaluate an empty test set.")

    parse_success = 0
    exact_match = 0
    schema_valid = 0
    schema_examples = 0
    field_correct: dict[str, int] = {}
    field_total: dict[str, int] = {}
    for index, (target, output) in enumerate(zip(expected, generated)):
        schema = schemas[index] if schemas is not None else None
        target = validate_json_object(target, schema)
        parsed = True
        try:
            prediction = decode_json_object(output)
        except (ValueError, TypeError):
            prediction = {}
            parsed = False
        else:
            parse_success += 1
        if schema is not None:
            schema_examples += 1
            if parsed:
                try:
                    validate_json_object(prediction, schema)
                except ValueError:
                    pass
                else:
                    schema_valid += 1
        exact_match += parsed and prediction == target
        for field in target.keys() | prediction.keys():
            field_total[field] = field_total.get(field, 0) + 1
            field_correct.setdefault(field, 0)
            if field in target and field in prediction and prediction[field] == target[field]:
                field_correct[field] = field_correct.get(field, 0) + 1

    count = len(expected)
    field_accuracy = {
        field: correct / field_total[field]
        for field, correct in sorted(field_correct.items())
    }
    field_value_accuracy = sum(field_correct.values()) / max(1, sum(field_total.values()))
    report = {
        "examples": count,
        "json_parse_rate": parse_success / count,
        "exact_match_accuracy": exact_match / count,
        "field_accuracy": field_accuracy,
        "field_value_accuracy": field_value_accuracy,
    }
    if schema_examples:
        report["schema_validation_examples"] = schema_examples
        report["schema_valid_rate"] = schema_valid / schema_examples
    return report
