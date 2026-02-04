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

### (Optional) Run Test Suite
While you have docker compose running in one terminal, in another run the following:
```bash
pytest -v -s tests/
```


## Testing It Out

After following the Developer Quickstart `docker compose up`, start the dummy customer api.

```bash
python dummy_customer_api.py
```

Navigate to [htttp://localhost:3000](htttp://localhost:3000) and ask your queries there!


## System Architecture
![system architecture](docs/sa_diagram.drawio.svg)