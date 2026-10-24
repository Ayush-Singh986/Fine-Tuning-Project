from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finetune.contracts import conversation_messages, parse_json_object, validate_schema
from finetune.model import Extractor, TransformersExtractor


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")


class ExtractResponse(BaseModel):
    result: dict[str, Any]


def create_app(extractor: Extractor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if extractor is not None:
            app.state.extractor = extractor
        else:
            app.state.extractor = TransformersExtractor.from_environment()
        yield

    app = FastAPI(
        title="Structured JSON Extractor",
        version="0.1.0",
        description="Return JSON for a request, optionally validated against a JSON Schema.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/extract", response_model=ExtractResponse)
    def extract(request: ExtractRequest) -> ExtractResponse:
        if request.json_schema is not None and len(str(request.json_schema)) > 20_000:
            raise HTTPException(
                status_code=422,
                detail="The supplied JSON Schema exceeds 20,000 characters.",
            )
        if request.json_schema is not None:
            try:
                validate_schema(request.json_schema)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            raw_output = app.state.extractor.generate(
                conversation_messages(request.text, request.json_schema)
            )
            result = parse_json_object(raw_output, request.json_schema)
        except ValueError as error:
            raise HTTPException(status_code=502, detail=f"Model returned invalid output: {error}") from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail="Model is not available.") from error
        return ExtractResponse(result=result)

    return app


app = create_app()
