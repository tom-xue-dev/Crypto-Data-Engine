# Repository Guidelines

## Project Structure & Module Organization
- `src/crypto_data_engine/`: FastAPI server, services, common config, DB access.
- `src/task_manager/`: Celery app and registered tasks (tick download, extract, health).
- `deploy/`: Docker Compose and deployment helpers.
- `docs/`: API docs and design notes.
- `data/`, `logs/`: Local artifacts; ignore large files in commits.
- `pyproject.toml`, `poetry.lock`: Dependencies and CLI entrypoints (Poetry).

## Build, Test, and Development Commands
- Install: `poetry install` (ensure Python 3.12). Optionally: `poetry shell`.
- Start API: `poetry run main start` (honors `.env` via settings).
- Celery worker: `poetry run main run-worker downloader` or `celery -A task_manager.celery_app worker --loglevel=info`.
- Init configs: `poetry run main init-config` generates YAML templates in `data/config/config_templates/`.
- Init DB: `poetry run main init-db` to create required tables.
- Tests: `poetry run pytest -q` (add tests under `tests/`).

## Coding Style & Naming Conventions
- Python/PEP8: 4-space indents, max line length ~100.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Type hints: Prefer explicit annotations; keep public functions typed.
- Logging: Use `logging` or `loguru`; avoid `print` in library code.
- Imports: Absolute from `src` root (CLI sets `PYTHONPATH=src`).

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio` for async endpoints/tasks.
- Layout: `tests/` mirrors `src/` (e.g., `tests/services/test_tick_download.py`).
- Naming: Files start with `test_`, tests use `test_*` functions and fixtures.
- Scope: Cover core services (tick download, extract), API routes, and config loading.
- Run examples: `poetry run pytest tests/services -q`.

## Commit & Pull Request Guidelines
- Commits: Short, imperative subject; include scope when helpful. Examples:
  - `add: bar aggregation service`
  - `fix(db): handle missing env vars`
- PRs: Clear description, linked issues, reproduction/run steps, and logs/screenshots for behavior changes.
- Checks: Ensure `pytest` passes; no large data/logs committed; keep diffs focused.

## Security & Configuration Tips
- `.env` controls runtime: `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `DB_URL`, `SERVER_HOST/PORT`, `TICK_*`.
- Do not commit secrets. Prefer local `.env` and override via environment in CI/deploy.
- Paths: Data defaults to `data/` (see `TickDownloadConfig`).
