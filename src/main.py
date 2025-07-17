from fastapi import FastAPI
from services.tick_service.router import router as tick_router
from services.feature_service.router import router as feature_router
from services.backtest_service.router import router as backtest_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="BTC Trading Platform",
        version="0.0.1",
        description="Modular FastAPI prototype with tick ingestion, feature generation and backtesting services."
    )
    app.include_router(tick_router, prefix="/tick", tags=["tick-service"])
    app.include_router(feature_router, prefix="/features", tags=["feature-service"])
    app.include_router(backtest_router, prefix="/backtest", tags=["backtest-service"])
    return app

app = create_app()