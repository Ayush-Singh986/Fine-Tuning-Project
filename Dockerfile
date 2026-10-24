FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[inference]"

ENV MODEL_PATH=/models/adapter
EXPOSE 8000

CMD ["uvicorn", "finetune.api:app", "--host", "0.0.0.0", "--port", "8000"]
