ARG REGISTRY_MIRROR=docker.io
FROM ${REGISTRY_MIRROR}/archlinux:base AS base

ARG PYPI_MIRROR

WORKDIR /app

RUN pacman -Syu --noconfirm python python-pip curl && \
    pacman -Scc --noconfirm

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN if [ -n "$PYPI_MIRROR" ]; then \
        pip install --no-cache-dir --break-system-packages -i "$PYPI_MIRROR" .; \
    else \
        pip install --no-cache-dir --break-system-packages .; \
    fi

EXPOSE 7860

ENTRYPOINT ["llama-wrangler"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
