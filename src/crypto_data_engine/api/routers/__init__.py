"""
API routers for the crypto data engine.
"""
from .aggregation import aggregation_router
from .asset_pool import asset_pool_router
from .backtest import router as backtest_router
from .data_inventory import data_inventory_router
from .download import download_router
from .feature import feature_router
from .strategy import router as strategy_router
from .visualization import router as visualization_router

__all__ = [
    "aggregation_router",
    "asset_pool_router",
    "backtest_router",
    "data_inventory_router",
    "download_router",
    "feature_router",
    "strategy_router",
    "visualization_router",
]
