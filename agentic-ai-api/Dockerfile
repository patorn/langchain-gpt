# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (add poppler-utils if doing PDF w/ some loaders)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY agentic-ai-api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agentic-ai-api/app ./app
COPY agentic-ai-api/frontend ./frontend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
