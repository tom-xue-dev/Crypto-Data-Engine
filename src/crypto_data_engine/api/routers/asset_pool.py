"""
Asset pool selection API routes.
"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

asset_pool_router = APIRouter(prefix="/api/v1/asset-pool", tags=["Asset Pool"])


class AssetPoolRequest(BaseModel):
    top_n: int = 100
    cache_dir: Optional[str] = None


@asset_pool_router.post("/select", summary="Select top N assets by volume")
async def select_asset_pool(request: AssetPoolRequest):
    """Run asset pool selection from Binance Futures API."""
    try:
        from crypto_data_engine.services.asset_pool.asset_selector import (
            AssetPoolConfig,
            AssetPoolSelector,
        )

        config = AssetPoolConfig(
            top_n=request.top_n,
            cache_dir=Path(request.cache_dir) if request.cache_dir else None,
        )
        selector = AssetPoolSelector(config)
        symbols = selector.get_current_pool(force_refresh=True)
        pool_info = selector.get_pool_info()

        return {
            "count": len(symbols),
            "symbols": symbols,
            "info": pool_info,
        }
    except Exception as error:
        logger.error(f"Asset pool selection failed: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@asset_pool_router.get("/current", summary="Get cached asset pool")
async def get_current_pool(
    cache_dir: Optional[str] = Query(None, description="Cache directory path"),
):
    """Return the most recently cached asset pool without refreshing."""
    try:
        from crypto_data_engine.services.asset_pool.asset_selector import (
            AssetPoolConfig,
            AssetPoolSelector,
        )

        config = AssetPoolConfig(
            cache_dir=Path(cache_dir) if cache_dir else None,
        )
        selector = AssetPoolSelector(config)
        symbols = selector.get_current_pool(force_refresh=False)

        return {
            "count": len(symbols),
            "symbols": symbols,
        }
    except Exception as error:
        logger.error(f"Failed to get current pool: {error}")
        raise HTTPException(status_code=500, detail=str(error))
