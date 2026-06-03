FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY lumen_rag ./lumen_rag
RUN pip install --no-cache-dir ".[openai]"

EXPOSE 8000
CMD ["lumen", "serve", "--host", "0.0.0.0", "--port", "8000"]
