# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libsecret-tools && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src /app/src/
COPY configs /app/configs/
COPY requirements.txt /app/requirements.txt
COPY requirements_dev.txt /app/requirements_dev.txt
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --verbose
RUN --mount=type=cache,target=/root/.cache/pip pip install . --no-deps --verbose

# Environment variable for wandb (to be overridden at runtime)
ENV WANDB_API_KEY=""

ENTRYPOINT ["python", "-u", "src/fruit_vegetable_classification/train.py"]
