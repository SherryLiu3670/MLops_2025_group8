# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY configs configs/
COPY models models/
COPY requirements_api.txt requirements_api.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_api.txt --verbose
RUN --mount=type=cache,target=/root/.cache/pip pip install . --no-deps --verbose

EXPOSE 8080

ENTRYPOINT ["uvicorn", "src.fruit_vegetable_classification.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
# uvicorn src.fruit_vegetable_classification.api:app --host 0.0.0.0 --port 8000
