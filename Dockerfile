ARG REGISTRY_MIRROR=docker.io
FROM ${REGISTRY_MIRROR}/python:3.12-slim AS base

ARG PYPI_MIRROR

WORKDIR /app

# Install system deps for nvidia-smi (if available in runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN if [ -n "$PYPI_MIRROR" ]; then \
        pip install --no-cache-dir -i "$PYPI_MIRROR" .; \
    else \
        pip install --no-cache-dir .; \
    fi

EXPOSE 7860

ENTRYPOINT ["llama-wrangler"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
