# DQ Swirl

Agentic Data Quality & Querying

## Developer Quickstart

Add a `secrets.env` file with the following contents at the root of the repo
```dotenv
LLM_BASE_URL="https://openrouter.ai/api/v1"
LLM_API_KEY=<Your API KEY>
```

Create virtual environment and install dependencies
```bash
uv venv
source .venv/bin/activate

uv tool install maturin
uv sync --all-groups --extra app --extra worker && maturin develop --release
```

Start the services
```
docker compose up
```

## Demo

After following the Developer Quickstart `docker compose up`, start the dummy customer api.

```bash
python app/dummy_customer_api.py
```

Navigate to [htttp://localhost:3000](htttp://localhost:3000) and ask your queries there!


## Run Test Suite
While you have docker compose running in one terminal, in another run the following:
```bash
pytest -v -s tests/
```


## System Architecture
![system architecture](docs/sa_diagram.drawio.svg)