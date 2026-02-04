FROM python:3.14-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"


# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-system

WORKDIR /app
RUN uv venv


COPY worker/ ./worker/
COPY src/ ./src/
COPY rust/ ./rust/
COPY pyproject.toml Cargo.toml README.md ./

RUN uv tool install maturin
ENV PATH="/root/.local/bin:$PATH" 

RUN uv sync --extra worker && maturin develop --release
RUN uv pip list

FROM python:3.14-slim AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/.venv/lib/python3.14/site-packages" \
    PYTHONUNBUFFERED=1

USER appuser

CMD ["saq", "worker.saq_worker.settings"]