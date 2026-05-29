# Official Python image
FROM python:3.11-slim

ENV POETRY_VERSION=1.8.2

WORKDIR /app

RUN pip3 install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* ./

COPY src ./src

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
