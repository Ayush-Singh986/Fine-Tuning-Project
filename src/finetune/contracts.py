# Import the standard JSON library for parsing and serializing JSON data
import json
# Import the regular expression module for pattern matching in text
import re
# Import the Any type hint for flexible type annotations
from typing import Any

# Import JSON Schema validation components from the jsonschema library
# Draft202012Validator: The validator class for JSON Schema Draft 2020-12
# FormatChecker: Validates format constraints like dates, emails, etc.
# SchemaError: Raised when a schema itself is invalid
from jsonschema import Draft202012Validator, FormatChecker, SchemaError

# Define the invoice schema as a constant
# This schema defines the expected structure for invoice JSON objects
# It serves as a contract for what constitutes a valid invoice extraction
INVOICE_SCHEMA: dict[str, Any] = {
    # Specify the JSON Schema version being used
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    # The root must be a JSON object (not array, string, etc.)
    "type": "object",
    # Reject any fields not explicitly defined in this schema
    # This prevents models from adding hallucinated fields
    "additionalProperties": False,
    # These fields MUST be present in every invoice
    "required": ["vendor", "invoice_number", "date", "total", "currency"],
    # Define the structure and constraints for each field
    "properties": {
        # Vendor name can be a string or null (if not found in document)
        "vendor": {"type": ["string", "null"]},
        # Invoice number can be a string or null
        "invoice_number": {"type": ["string", "null"]},
        # Date must follow YYYY-MM-DD format if provided
        "date": {
            "type": ["string", "null"],
            # Regex pattern enforces YYYY-MM-DD format
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
            # The "format" keyword enables semantic validation
            "format": "date",
        },
        # Total amount must be a number or null
        "total": {"type": ["number", "null"]},
        # Currency must be a 3-letter ISO code (e.g., USD, EUR) or null
        "currency": {
            "type": ["string", "null"],
            # Enforce exactly 3 uppercase letters
            "pattern": r"^[A-Z]{3}$",
        },
    },
}

# Create a reusable validator instance for the invoice schema
# Pre-creating this improves performance when validating multiple invoices
# FormatChecker() enables validation of format keywords like "date"
_INVOICE_VALIDATOR = Draft202012Validator(
    INVOICE_SCHEMA,
    format_checker=FormatChecker(),
)

# Define the system prompt that instructs the model how to behave
# This prompt is critical for consistent model behavior during inference
# It sets expectations for JSON output format and handling of missing data
SYSTEM_PROMPT = (
    # Instruct the model to follow user instructions precisely
    "Follow the user's instructions and preserve any schema or format they "
    "provide. When the user requests JSON, return one valid JSON object only, "
    # Prevent the model from wrapping JSON in markdown code blocks
    # This eliminates post-processing to extract JSON from markdown
    "without markdown or commentary. Do not invent facts; use null or an "
    # Instruct the model to explicitly handle missing information
    # This prevents hallucination when data is not present in the source text
    "explicit error field when the instructions require missing information "
    "to be represented."
)


def make_prompt(text: str, schema: dict[str, Any] | None = None) -> str:
    """Build a complete prompt string for models without chat template support.
    
    This function combines the system instructions, user request, and optional
    schema into a single prompt string. Use this for models that don't support
    the modern chat template format (older models or simple fine-tuning setups).
    
    Args:
        text: The user's extraction request or document text
        schema: Optional JSON Schema to validate the output against
        
    Returns:
        A complete prompt string ready for tokenization
    """
    # Start with the system instructions and user request
    # The double newline creates clear visual separation
    prompt = f"{SYSTEM_PROMPT}\n\nUser request and context:\n{text}"
    # If a schema is provided, append it so the model knows the expected structure
    # ensure_ascii=False preserves Unicode characters (important for international invoices)
    if schema is not None:
        prompt += "\n\nRequired JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
    return prompt


def conversation_messages(
    text: str,
    schema: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build a conversation in the standard chat format for models with chat templates.
    
    Modern language models use a structured chat format with role/content pairs.
    This format is processed by the tokenizer's chat_template to create the
    model-specific prompt format (e.g., ChatML, Alpaca, etc.).
    
    Args:
        text: The user's extraction request or document text
        schema: Optional JSON Schema to validate the output against
        
    Returns:
        A list of message dictionaries with 'role' and 'content' keys
    """
    # Start with the user's text
    user_content = text
    # If a schema is provided, append it to the user message
    # This ensures the schema is part of the user's request, not system instructions
    if schema is not None:
        user_content += (
            "\n\nRequired JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    # Return the conversation in standard chat format
    # System message sets behavior, user message provides the task
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_json_object(
    text: str,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse and validate a JSON object from model output.
    
    This is a convenience function that combines decoding and validation.
    Use this when you need both steps in a single call.
    
    Args:
        text: Raw text output from the model (may contain markdown)
        schema: Optional JSON Schema to validate against
        
    Returns:
        A validated JSON object (dictionary)
        
    Raises:
        ValueError: If the text is not valid JSON or doesn't match the schema
    """
    # First, extract the JSON from the text (handles markdown code blocks)
    value = decode_json_object(text)
    # Then validate it against the schema if one is provided
    validate_json_object(value, schema)
    return value


def decode_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from model output, allowing a single fenced JSON block.
    
    Models sometimes wrap JSON in markdown code blocks (```json ... ```).
    This function extracts the JSON content from such blocks before parsing.
    This makes the parser robust to both raw JSON and markdown-wrapped JSON.
    
    Args:
        text: Raw text output from the model
        
    Returns:
        A JSON object (dictionary)
        
    Raises:
        ValueError: If the text is not valid JSON or not an object
        json.JSONDecodeError: If JSON parsing fails
    """
    # Remove leading/trailing whitespace from the model's response
    response = text.strip()
    # Check if the response is wrapped in a markdown code block
    # Pattern matches ```json or ``` followed by content and closing ```
    # re.DOTALL allows . to match newlines (multi-line JSON)
    # re.IGNORECASE allows both ```json and ```JSON
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    # If it's fenced, extract just the JSON content inside the code block
    if fenced:
        response = fenced.group(1)
    # Parse the JSON string into a Python object
    value = json.loads(response)
    # Ensure the parsed value is a JSON object (dictionary), not an array or primitive
    # This is critical because our invoice extraction expects object output
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object.")
    return value


def validate_target(target: Any) -> dict[str, Any]:
    """Validate that a target value matches the invoice schema.
    
    This function is used during dataset preparation to ensure that
    all training examples have valid invoice targets. This prevents
    training on malformed data.
    
    Args:
        target: The target value to validate (should be a dict)
        
    Returns:
        The validated target (if validation passes)
        
    Raises:
        ValueError: If target is not a dict or doesn't match the invoice schema
    """
    # Ensure the target is a JSON object (dictionary)
    if not isinstance(target, dict):
        raise ValueError("target must be a JSON object.")
    # Validate against the invoice schema and collect all errors
    # iter_errors returns all validation errors (not just the first one)
    errors = sorted(
        _INVOICE_VALIDATOR.iter_errors(target),
        # Sort errors by their path in the JSON structure for consistent error messages
        key=lambda error: list(error.path),
    )
    # If there are validation errors, raise an exception with details
    if errors:
        # Join all error messages with semicolons for a comprehensive error report
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"target does not match the invoice schema: {details}")
    return target


def validate_json_object(
    target: Any,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a JSON object against an optional JSON Schema.
    
    This is a general-purpose validation function that works with any schema.
    Use this when you need to validate against a custom schema (not the invoice schema).
    
    Args:
        target: The JSON object to validate
        schema: Optional JSON Schema to validate against
        
    Returns:
        The validated target (if validation passes)
        
    Raises:
        ValueError: If target is not a dict or doesn't match the schema
    """
    # Ensure the target is a JSON object (dictionary)
    if not isinstance(target, dict):
        raise ValueError("target must be a JSON object.")
    # If no schema is provided, just return the target (no validation needed)
    if schema is None:
        return target
    # Validate the schema itself before using it
    # This prevents cryptic errors during validation
    validate_schema(schema)
    # Create a validator for the provided schema
    # FormatChecker enables validation of format keywords (date, email, etc.)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    # Validate the target and collect all errors
    errors = sorted(validator.iter_errors(target), key=lambda error: list(error.path))
    # If there are validation errors, raise an exception with details
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"JSON object does not match the supplied schema: {details}")
    return target


def validate_schema(schema: dict[str, Any]) -> None:
    """Validate that a JSON Schema is correctly formed.
    
    Before using a schema to validate data, we must validate the schema itself.
    This catches errors in schema definitions early and provides clear error messages.
    
    Args:
        schema: The JSON Schema to validate
        
    Raises:
        ValueError: If the schema is invalid
    """
    try:
        # Check that the schema is valid according to JSON Schema Draft 2020-12
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        # Re-raise as ValueError for consistent error handling in our application
        raise ValueError(f"Invalid JSON Schema: {error.message}") from error
