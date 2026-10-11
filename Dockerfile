FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY lumen_rag ./lumen_rag
RUN pip install --no-cache-dir ".[openai]"

# Run unprivileged. The app writes its index (.lumen_index) under /app, so
# the user owns the workdir; uid 1000 also matches Hugging Face Docker Spaces.
RUN useradd --create-home --uid 1000 app && chown app:app /app
USER app

EXPOSE 8000
CMD ["lumen", "serve", "--host", "0.0.0.0", "--port", "8000"]
