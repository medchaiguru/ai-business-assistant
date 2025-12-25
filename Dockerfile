FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Install dependencies from lock file into system python
# 1. Export lock file to requirements.txt (excluding dev deps)
# 2. Sync system python to match requirements.txt
RUN uv export --format requirements-txt --no-dev --frozen > requirements.txt && \
    uv pip sync --system requirements.txt

# Copy app code
COPY app/ ./app/
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]