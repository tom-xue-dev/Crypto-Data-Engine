# BTC Trading FastAPI Scaffold

Minimal micro-service style scaffold for your quantitative trading platform.

## Services
| Service | Mounted Prefix | Responsibilities |
|---------|----------------|------------------|
| tick-service | `/tick` | Ingest and serve raw tick or aggregated bars |
| feature-service | `/features` | Generate and cache technical factors / alphas |
| backtest-service | `/backtest` | Run strategy backtests asynchronously |

## Quick Start (local mono‑repo)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# then open http://127.0.0.1:8000/docs
```

## Next Steps
1. Implement database dependencies (TimescaleDB / Redis).
2. Replace placeholders in routers with actual logic.
3. Split each service into its own Dockerfile & container for Compose.
4. Add adapters for caching and parallel computation (Redis, Ray).