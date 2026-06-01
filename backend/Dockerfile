FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    linux-libc-dev \
    tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry --no-cache-dir

COPY pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry lock \
    && poetry install --no-interaction --no-ansi --no-root

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
