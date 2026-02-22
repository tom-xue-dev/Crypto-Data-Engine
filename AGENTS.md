# Repository Guidelines

## Project Structure & Module Organization
Core backend code lives in `src/crypto_data_engine/`:
- `app/`: Typer CLI commands (`serve`, `data`, `aggregate`, `backtest`, `pipeline`)
- `api/`: FastAPI app, routers, and Pydantic schemas
- `services/`: domain logic (tick download, bar aggregation, features, signals, backtest)
- `common/` and `core/`: shared config, logging, task management, interfaces, base types

Research/experiment scripts are in `scripts/` (many `run_*.py` files). Tests are in `tests/`. Deployment assets are in `deploy/` (`Dockerfile`, `docker-compose.yml`). Frontend code is in `frontend/` (Vite + React + TypeScript).

## Build, Test, and Development Commands
- `poetry install`: install backend dependencies.
- `poetry run main dev`: run API locally with auto-reload on `127.0.0.1:8000`.
- `poetry run main serve --host 127.0.0.1 --port 8000`: run production-style API.
- `poetry run main pipeline run --top-n 100 --threshold 1h --workers 4`: run full research pipeline.
- `poetry run main test`: run all backend tests.
- `poetry run main test --file test_bar_aggregator.py`: run one test file.
- `poetry run main test --coverage`: run tests with coverage output.
- `cd frontend && npm install && npm run dev`: run frontend at `http://localhost:5173`.

## Coding Style & Naming Conventions
Target Python is 3.12. Follow existing PEP 8 style with 4-space indentation and type hints for public interfaces. Use:
- `snake_case` for modules/functions/variables
- `PascalCase` for classes
- clear, domain-specific names (`AssetPoolSelector`, `aggregate_bars`)

Keep imports grouped (stdlib, third-party, local) and keep CLI-facing behavior in `app/` rather than service modules.

## Testing Guidelines
Use `pytest` and `pytest-asyncio`. Place tests under `tests/` with `test_*.py` names and `test_*` functions/classes. Prefer deterministic synthetic fixtures for unit tests; isolate slower, data-dependent checks with markers (for example `@pytest.mark.slow`). Some integration tests expect local data roots such as `E:/data`; skip gracefully when unavailable.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects (for example `add bar aggregation service`, `update worker communication`). Keep commits focused to one logical change. For PRs, include:
- what changed and why
- impacted modules/paths
- test evidence (commands run and key results)
- screenshots for frontend or visualization changes
- linked issue/task when available

## Security & Configuration Tips
Use `.env` for secrets and never commit credentials. Keep generated datasets, reports, and logs out of Git unless intentionally versioned. Validate data/config paths before running long pipeline jobs.
