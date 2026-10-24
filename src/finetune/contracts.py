import json
import re
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, SchemaError

INVOICE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["vendor", "invoice_number", "date", "total", "currency"],
    "properties": {
        "vendor": {"type": ["string", "null"]},
        "invoice_number": {"type": ["string", "null"]},
        "date": {
            "type": ["string", "null"],
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
            "format": "date",
        },
        "total": {"type": ["number", "null"]},
        "currency": {
            "type": ["string", "null"],
            "pattern": r"^[A-Z]{3}$",
        },
    },
}
_INVOICE_VALIDATOR = Draft202012Validator(
    INVOICE_SCHEMA,
    format_checker=FormatChecker(),
)

SYSTEM_PROMPT = (
    "Follow the user's instructions and preserve any schema or format they "
    "provide. When the user requests JSON, return one valid JSON object only, "
    "without markdown or commentary. Do not invent facts; use null or an "
    "explicit error field when the instructions require missing information "
    "to be represented."
)


def make_prompt(text: str, schema: dict[str, Any] | None = None) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nUser request and context:\n{text}"
    if schema is not None:
        prompt += "\n\nRequired JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
    return prompt


def conversation_messages(
    text: str,
    schema: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    user_content = text
    if schema is not None:
        user_content += (
            "\n\nRequired JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_json_object(
    text: str,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = decode_json_object(text)
    validate_json_object(value, schema)
    return value


def decode_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object, allowing a single fenced JSON block."""
    response = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        response = fenced.group(1)
    value = json.loads(response)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object.")
    return value


def validate_target(target: Any) -> dict[str, Any]:
    if not isinstance(target, dict):
        raise ValueError("target must be a JSON object.")
    errors = sorted(
        _INVOICE_VALIDATOR.iter_errors(target),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"target does not match the invoice schema: {details}")
    return target


def validate_json_object(
    target: Any,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(target, dict):
        raise ValueError("target must be a JSON object.")
    if schema is None:
        return target
    validate_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(target), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"JSON object does not match the supplied schema: {details}")
    return target


def validate_schema(schema: dict[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError(f"Invalid JSON Schema: {error.message}") from error
