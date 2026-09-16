ARG REGISTRY_MIRROR=docker.io
FROM ${REGISTRY_MIRROR}/archlinux:base AS base

ARG PYPI_MIRROR
ARG EXTRA_INDEX_URL
ARG LOCAL_WHEEL
ARG PACKAGE_VERSION

WORKDIR /app

RUN pacman -Syu --noconfirm python python-pip curl && \
    pacman -Scc --noconfirm

RUN mkdir -p /tmp/dist/
COPY dist/ /tmp/dist/

RUN PIP_FLAGS="--no-cache-dir --break-system-packages"; \
    if [ -n "$PYPI_MIRROR" ]; then \
        PIP_FLAGS="$PIP_FLAGS -i $PYPI_MIRROR"; \
    fi; \
    if [ -n "$LOCAL_WHEEL" ] && [ -f "/tmp/dist/$LOCAL_WHEEL" ]; then \
        echo "Installing from local wheel: $LOCAL_WHEEL"; \
        pip install $PIP_FLAGS "/tmp/dist/$LOCAL_WHEEL"; \
    elif [ -n "$EXTRA_INDEX_URL" ] && [ -n "$PACKAGE_VERSION" ]; then \
        # Step 1: install the package from TestPyPI (no deps to avoid broken transitives)
        echo "Installing llama-wrangler==$PACKAGE_VERSION from $EXTRA_INDEX_URL (no-deps)"; \
        pip install $PIP_FLAGS --extra-index-url "$EXTRA_INDEX_URL" --no-deps \
            "llama-wrangler==$PACKAGE_VERSION"; \
        # Step 2: let pip resolve dependencies from regular PyPI
        echo "Installing dependencies from PyPI"; \
        pip install $PIP_FLAGS --extra-index-url "$EXTRA_INDEX_URL" \
            "llama-wrangler==$PACKAGE_VERSION"; \
    elif [ -n "$PACKAGE_VERSION" ]; then \
        echo "Installing llama-wrangler==$PACKAGE_VERSION"; \
        pip install $PIP_FLAGS "llama-wrangler==$PACKAGE_VERSION"; \
    else \
        echo "Installing latest llama-wrangler from PyPI"; \
        pip install $PIP_FLAGS "llama-wrangler"; \
    fi && \
    rm -rf /tmp/dist/

EXPOSE 7860

ENTRYPOINT ["llama-wrangler"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
