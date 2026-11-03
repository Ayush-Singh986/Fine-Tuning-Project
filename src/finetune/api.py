# Import asynccontextmanager for managing application lifecycle
from contextlib import asynccontextmanager
# Import Any for flexible type annotations
from typing import Any

# Import FastAPI framework components for building REST APIs
from fastapi import FastAPI, HTTPException
# Import Pydantic for request/response validation and serialization
from pydantic import BaseModel, Field

# Import our helper functions for prompt construction and validation
from finetune.contracts import conversation_messages, parse_json_object, validate_schema
# Import the Extractor protocol and implementation
from finetune.model import Extractor, TransformersExtractor


class ExtractRequest(BaseModel):
    """Request model for the /extract endpoint.
    
    Pydantic models provide automatic validation and documentation.
    The model validates that:
    - text is between 1 and 20,000 characters
    - json_schema (if provided) is a valid JSON object
    
    The Field() function allows us to specify validation rules and Swagger UI metadata.
    """
    # The input text to extract structured data from
    # min_length=1 ensures the text is not empty
    # max_length=20_000 prevents extremely long inputs that could crash the model
    text: str = Field(min_length=1, max_length=20_000)
    # Optional JSON Schema to validate the output against
    # alias="schema" allows clients to use "schema" instead of "json_schema"
    # This makes the API more intuitive for users
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")


class ExtractResponse(BaseModel):
    """Response model for the /extract endpoint.
    
    This defines the structure of the API response.
    FastAPI automatically validates the response matches this structure
    and generates OpenAPI documentation.
    """
    # The extracted JSON object
    result: dict[str, Any]


def create_app(extractor: Extractor | None = None) -> FastAPI:
    """Create and configure a FastAPI application for JSON extraction.
    
    This factory function creates a new FastAPI app instance.
    Using a factory instead of a global app allows for:
    - Easier testing with mock extractors
    - Multiple app instances with different configurations
    - Dependency injection of the extractor
    
    Args:
        extractor: Optional extractor instance (if None, loads from environment)
        
    Returns:
        A configured FastAPI application
    """
    # Define the application lifecycle using an async context manager
    # This handles startup and shutdown events
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle (startup and shutdown).
        
        This is called when the application starts and stops.
        We use it to initialize the model and store it in app.state.
        """
        # If an extractor was provided, use it directly
        if extractor is not None:
            app.state.extractor = extractor
        # Otherwise, load the extractor from environment variables
        # This is useful for production deployment
        else:
            app.state.extractor = TransformersExtractor.from_environment()
        # Yield control to the application (it's now running)
        yield
        # After yield, the application is shutting down
        # Add cleanup code here if needed (e.g., freeing GPU memory)

    # Create the FastAPI application with metadata
    # This metadata appears in the Swagger UI documentation
    app = FastAPI(
        title="Structured JSON Extractor",
        version="0.1.0",
        description="Return JSON for a request, optionally validated against a JSON Schema.",
        lifespan=lifespan,  # Attach the lifecycle manager
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check endpoint for monitoring and load balancers.
        
        Kubernetes, Docker, and cloud load balancers use health checks
        to determine if the service is running properly.
        A simple 200 OK response indicates the service is healthy.
        """
        return {"status": "ok"}

    @app.post("/extract", response_model=ExtractResponse)
    def extract(request: ExtractRequest) -> ExtractResponse:
        """Extract structured JSON from unstructured text.
        
        This is the main endpoint that performs JSON extraction.
        It:
        1. Validates the input schema (if provided)
        2. Generates a response from the model
        3. Parses and validates the response
        4. Returns the structured JSON
        
        Args:
            request: The extraction request containing text and optional schema
            
        Returns:
            ExtractResponse containing the extracted JSON object
            
        Raises:
            HTTPException: If validation fails or the model errors
        """
        # Validate schema size to prevent DoS attacks
        # Large schemas could consume excessive memory during validation
        if request.json_schema is not None and len(str(request.json_schema)) > 20_000:
            raise HTTPException(
                status_code=422,
                detail="The supplied JSON Schema exceeds 20,000 characters.",
            )
        
        # Validate the schema structure before using it
        # This provides a clear error message for malformed schemas
        if request.json_schema is not None:
            try:
                validate_schema(request.json_schema)
            except ValueError as error:
                # Return 422 Unprocessable Entity for invalid input
                raise HTTPException(status_code=422, detail=str(error)) from error
        
        # Generate and parse the model's response
        try:
            # Generate the raw text response from the model
            # conversation_messages creates the prompt with system/user messages
            raw_output = app.state.extractor.generate(
                conversation_messages(request.text, request.json_schema)
            )
            # Parse the JSON and validate against the schema (if provided)
            result = parse_json_object(raw_output, request.json_schema)
        except ValueError as error:
            # The model returned invalid JSON or validation failed
            # Return 502 Bad Gateway to indicate an upstream (model) error
            raise HTTPException(status_code=502, detail=f"Model returned invalid output: {error}") from error
        except RuntimeError as error:
            # The model failed to load or is otherwise unavailable
            # Return 503 Service Unavailable
            raise HTTPException(status_code=503, detail="Model is not available.") from error
        
        # Return the successful result
        return ExtractResponse(result=result)

    # Return the configured app
    return app


# Create a global app instance for easy deployment
# This is used when running: uvicorn finetune.api:app
app = create_app()
