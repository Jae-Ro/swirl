# DQ Swirl

Agentic Data Quality & Querying

## Developer Quickstart

```bash
uv venv
source .venv/bin/activate

uv tool install maturin
uv sync --all-groups --extra app --extra worker && maturin develop --release

docker compose up
```