# Official slim Python image
FROM python:3.11-slim

# Environment
ENV POETRY_VERSION=1.8.2

WORKDIR /app

# Install Poetry via pip
RUN pip3 install "poetry==$POETRY_VERSION"

# Dependency manifests
COPY pyproject.toml poetry.lock* ./

# FastAPI source and bundled models
COPY src ./src
COPY models ./models

# Install dependencies (no virtualenv inside container)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python3", "src/predictor/main.py"]
