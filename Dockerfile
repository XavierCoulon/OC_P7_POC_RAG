FROM python:3.12-slim

WORKDIR /app

# Set HuggingFace cache directory to a persistent location
ENV HF_HOME=/app/data/hf_cache
ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-install-project

# Copy project files
COPY . .

# Sync the project
RUN uv sync --frozen

# Pre-download HuggingFace model for faster container startup
# This ensures the model is cached in the image
RUN mkdir -p /app/data/hf_cache && \
    python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', encode_kwargs={'normalize_embeddings': True})" || true

# Activate the venv in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run API
CMD ["python", "main.py"]
