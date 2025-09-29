# Crypto Data Engine

A modular framework for downloading cryptocurrency tick data, aggregating it into research-ready bars, and running strategy backtests. The stack combines a FastAPI API surface, Celery-based distributed workers, and a structured configuration system so you can orchestrate data pipelines end-to-end.

---
## Highlights
- **FastAPI service** exposing download and aggregation endpoints under `/api/v1`
- **Celery workers** for IO-heavy (tick download) and CPU-heavy (bar generation) tasks
- **Composable configuration** via Pydantic settings + YAML templates
- **Database persistence** for task state (PostgreSQL via SQLAlchemy)
- **CLI tooling** powered by Typer for local orchestration and setup
- **Docker support** for Redis, Postgres, API, and worker processes

---
## Architecture
```
Client / CLI
    ↓
FastAPI (`crypto_data_engine.server`)
    ↓ submit tasks
Celery (Redis broker & backend)
    ↓ dispatch queues (`io_intensive`, `cpu`)
Task workers (`task_manager.celery_worker`)
    ├─ Tick download pipeline
    └─ Bar aggregation pipeline
```
Optional services such as Flower (Celery monitoring) and a Ray cluster can be added when you need extra observability or distributed compute.

---
## Project Structure
```text
crypto-data-engine/
├── deploy/                  # Docker files and compose stack
├── docs/                    # API reference and design notes
├── src/
│   ├── crypto_data_engine/
│   │   ├── common/          # Config loaders, logging utilities
│   │   ├── db/              # Models, repositories, session helpers
│   │   ├── server/          # FastAPI app, routers, request/response schemas
│   │   └── services/        # Tick downloader, bar aggregator, backtest modules
│   └── task_manager/
│       ├── celery_app.py    # Celery configuration
│       └── celery_worker.py # Registered Celery tasks
├── data/                    # Local data artifacts (tick, aggregated, configs)
├── logs/                    # Log output (Loguru)
├── pyproject.toml           # Poetry project definition
└── README.md
```
> The `data/` directory is intended for local artifacts and is not meant to be checked into version control.

---
## Prerequisites
- Python **3.12**
- [Poetry](https://python-poetry.org/)
- Redis (broker/result backend for Celery)
- PostgreSQL (task metadata store)
- Optional: Docker & Docker Compose

---
## Setup
```bash
# clone the repo
git clone <repo-url>
cd crypto-data-engine

# install dependencies
poetry install

# activate virtual environment (optional)
poetry shell
```

### Environment Variables
Create a `.env` in the project root or export the variables in your shell:
```
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
DB_URL=postgresql+psycopg://admin:123456@localhost:5432/quantdb
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```
Adjust Redis/Postgres credentials to match your environment. Additional configuration values live under `data/config/config_templates/` and can be generated with the CLI command below.

---
## CLI Commands
All commands are exposed through Typer (`crypto_data_engine.main`). Prefix them with `poetry run` if you are not inside a Poetry shell.

- `poetry run main start [--host HOST] [--port PORT]`
  - Launch the FastAPI server (defaults pulled from `ServerConfig`).

- `poetry run main run-worker <module>`
  - Start a Celery worker bound to `task_manager.celery_app`. The `module` argument is informational; the worker currently consumes all registered queues.

- `poetry run main init-config`
  - Generate YAML configuration templates under `data/config/config_templates/` based on the Pydantic settings classes.

- `poetry run main init-db`
  - Initialize database tables required for task tracking.

- `poetry run main dev-all`
  - Convenience command that spins up three Celery workers (downloader, bar generator, backtest) plus a development FastAPI server. Use only in local environments.

---
## Running the Services
1. **Ensure infrastructure is available**
   - Redis (for Celery) and PostgreSQL (for task metadata)
2. **Initialize configs and DB**
   ```bash
   poetry run main init-config
   poetry run main init-db
   ```
3. **Start FastAPI**
   ```bash
   poetry run main start
   ```
4. **Start Celery worker(s)**
   ```bash
   poetry run main run-worker downloader
   ```
   The worker registers tasks such as `tick.download`, `tick.extract_task`, and `bar.aggregate`. You can run multiple workers and pin them to specific queues via Celery configuration if needed.

---
## API Usage
### Download Symbols
```
curl "http://localhost:8080/api/v1/download/exchanges"
```

### Trigger a Download Job
```
curl -X POST "http://localhost:8080/api/v1/download/downloads/jobs" \
  -H "Content-Type: application/json" \
  -d '{
        "exchange": "binance",
        "symbols": ["BTCUSDT"],
        "year": 2023,
        "months": [1, 2]
      }'
```
This call creates task entries in the database and enqueues Celery jobs (`tick.download`).

### Aggregate Bars
```
curl -X POST "http://localhost:8080/api/v1/aggregate/bars" \
  -H "Content-Type: application/json" \
  -d '{
        "exchange": "binance",
        "bar_type": "volume_bar",
        "threshold": 1000,
        "symbols": ["BTCUSDT"]
      }'
```
The API resolves defaults from `AggregationConfig`, pushes a `bar.aggregate` Celery task, and writes aggregation results under `data/data_aggrate/`.

---
## Data Layout
- `data/tick_data/` – Downloaded and processed tick data (per exchange/symbol)
- `data/tick_test/` – Sample archives used for local testing
- `data/data_aggrate/` – Generated bar files grouped by bar type
- `data/config/` – YAML configuration templates and overrides

You can change default locations via the settings classes in `crypto_data_engine.common.config`.

---
## Development Notes
- Logging is handled by Loguru. Logs are written to `logs/app.log` (see `crypto_data_engine.common.logger`).
- Database models and repositories live under `crypto_data_engine.db`.
- Celery task routing is configured in `task_manager.celery_app`.

### Testing
Pytest is configured in `pyproject.toml`. Once tests are added under `tests/`, run:
```bash
poetry run pytest -q
```

---
## Docker Compose
A reference stack is provided in `deploy/docker-compose.yml`.
```bash
# build and start services in detached mode
cd deploy
docker compose up -d
```
This brings up Redis, Postgres, the API service, a Celery worker, and Flower (Celery dashboard). Bind mounts map the source tree into containers for iterative development.

---
## Contributing
1. Fork the repository and create a feature branch.
2. Keep pull requests focused and include relevant documentation updates.
3. Run linting/tests before submitting (see `pyproject.toml` for available tooling).

---
## License
Released under the MIT License. See `LICENSE` (or project metadata) for details.
