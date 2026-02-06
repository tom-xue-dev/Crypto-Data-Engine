"""
FastAPI main application for crypto data engine backtesting system.

Usage:
    uvicorn crypto_data_engine.api.main:app --reload --port 8000

Or run directly:
    python -m crypto_data_engine.api.main
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import (
    aggregation_router,
    asset_pool_router,
    backtest_router,
    data_inventory_router,
    download_router,
    feature_router,
    strategy_router,
    visualization_router,
)
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting crypto-data-engine backtest API...")
    yield
    # Shutdown
    logger.info("Shutting down crypto-data-engine backtest API...")


def create_app(
    title: str = "Crypto Data Engine Backtest API",
    version: str = "1.0.0",
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        version: API version
        cors_origins: List of allowed CORS origins

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        version=version,
        description="""
## Crypto Data Engine Backtest API

A comprehensive backtesting system for cryptocurrency trading strategies.

### Features

- **Cross-sectional backtesting**: Fixed period rebalancing (e.g., weekly)
- **Time-series backtesting**: Per-bar decision making
- **Multi-asset backtesting**: Non-aligned dollar bars with portfolio tracking
- **Risk management**: Stop-loss strategies, drawdown limits, leverage control
- **Cost modeling**: Commission, slippage, funding fees, leverage fees

### Endpoints

- `/api/backtest/*` - Run and manage backtests
- `/api/strategy/*` - Strategy configuration and validation
- `/api/viz/*` - Visualization data for charts

### Quick Start

```python
import requests

# Submit a backtest
response = requests.post(
    "http://localhost:8000/api/backtest/run",
    json={
        "strategy": {"name": "momentum", "params": {"long_count": 10}},
        "mode": "cross_sectional",
        "initial_capital": 1000000,
    }
)
task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/api/backtest/status/{task_id}")

# Get results
result = requests.get(f"http://localhost:8000/api/backtest/result/{task_id}")
```
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",  # React dev server
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(backtest_router, prefix="/api")
    app.include_router(strategy_router, prefix="/api")
    app.include_router(visualization_router, prefix="/api")
    app.include_router(download_router)
    app.include_router(aggregation_router)
    app.include_router(data_inventory_router)
    app.include_router(asset_pool_router)
    app.include_router(feature_router)

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"},
        )

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": version}

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": title,
            "version": version,
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "endpoints": {
                "backtest": "/api/backtest",
                "strategy": "/api/strategy",
                "visualization": "/api/viz",
            },
        }

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "crypto_data_engine.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
