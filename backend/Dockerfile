# Multi-stage Dockerfile using uv (fast Python package manager)
# Stage 1: builder — install deps
# Stage 2: runtime — minimal image with only what we need

# === STAGE 1: Builder ===
FROM python:3.12-slim AS builder

# Install uv (fast package manager — our differentiator vs other teams)
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# Copy only dependency files first (for Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtualenv inside the image
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# Install the exact versions pinned in uv.lock (core deps + the `embeddings`
# group, model2vec, so the UC9 website-drift comparator works). Installing from
# the lock rather than pyproject's `>=` ranges keeps fresh builds reproducible —
# an unpinned sqlmodel upgrade once broke startup seeding. model2vec is
# pure-NumPy (no torch), keeping the image lean.
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv export --frozen --no-dev --group embeddings --no-emit-project -o /tmp/requirements.txt && \
    uv pip install --no-cache -r /tmp/requirements.txt

# Pre-download the model2vec static model into a baked HF cache so the live
# website-drift demo runs fully offline (no HuggingFace fetch at runtime).
ENV HF_HOME=/opt/hf_cache
RUN . /opt/venv/bin/activate && \
    python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"

# === STAGE: Test ===
# Runs the pytest suite (unit + ASGI integration). Reuses the full app venv and
# adds pytest/pytest-asyncio. Code + tests are bind-mounted at runtime by the
# `backend-tests` compose service, so edits run without rebuilding.
FROM builder AS test
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app
RUN . /opt/venv/bin/activate && \
    uv pip install --no-cache pytest pytest-asyncio pytest-bdd
CMD ["pytest", "-v"]

# === STAGE 2: Runtime ===
FROM python:3.12-slim AS runtime

# System deps for ML libraries (libgomp for XGBoost, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv + the baked model2vec/HF model cache from the builder so the
# website-drift comparator runs offline.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf_cache /opt/hf_cache
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf_cache

# Non-root user (security best practice)
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy app code
COPY --chown=app:app . .

# Make sure data dirs exist.
# data/db is owned by `app` so the named volume mounted there inherits
# writable ownership (SQLite needs to create the .db file at runtime).
RUN mkdir -p data/models data/mock data/audio_samples data/db

EXPOSE 8000

# Healthcheck — Docker uses this to know if container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
# --reload disabled in container (use volume mounts + reload in dev)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
