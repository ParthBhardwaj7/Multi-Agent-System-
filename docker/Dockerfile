# Multi-stage production image: small, non-root, no secrets, proper signals.

# --- Stage 1: dependencies -------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
RUN pip install --prefix=/install ".[dev]" 2>/dev/null || pip install --prefix=/install .

# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /srv appuser

WORKDIR /srv

# Dependencies from builder
COPY --from=builder /install /usr/local

# Application code (no .env, no secrets — config comes from the environment)
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY migrations/ ./migrations/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY pyproject.toml README.md ./

# Install the app package itself (without re-resolving dependencies)
RUN pip install --no-deps . \
    && python -c "import app.main" \
    && chmod -R a+rX /srv

USER appuser

EXPOSE 8000

# Graceful shutdown: uvicorn handles SIGTERM; exec form ensures PID 1 signal delivery.
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-graceful-shutdown", "20"]
