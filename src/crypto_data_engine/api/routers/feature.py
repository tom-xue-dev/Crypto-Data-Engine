"""
Feature calculation API routes.

Uses TaskManager for async execution.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.common.task_manager import get_task_manager

logger = get_logger(__name__)

feature_router = APIRouter(prefix="/api/v1/features", tags=["Feature Calculation"])


class FeatureCalculateRequest(BaseModel):
    bar_dir: str = "E:/data/bar_data"
    symbols: Optional[List[str]] = None
    windows: List[int] = [5, 10, 20, 60]
    include_alphas: bool = False
    include_technical: bool = False


def _run_feature_calculation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute feature calculation in a worker process."""
    from pathlib import Path

    import pandas as pd

    from crypto_data_engine.services.feature.unified_features import (
        UnifiedFeatureCalculator,
        UnifiedFeatureConfig,
    )

    bar_dir = Path(payload["bar_dir"])
    symbols = payload.get("symbols")
    windows = payload.get("windows", [5, 10, 20, 60])

    config = UnifiedFeatureConfig(
        windows=windows,
        include_returns=True,
        include_volatility=True,
        include_momentum=True,
        include_volume=True,
        include_microstructure=True,
        include_alphas=payload.get("include_alphas", False),
        include_technical=payload.get("include_technical", False),
        include_cross_sectional=False,
    )
    calculator = UnifiedFeatureCalculator(config)

    # Discover symbol directories
    if symbols:
        symbol_dirs = [bar_dir / s for s in symbols if (bar_dir / s).exists()]
    else:
        symbol_dirs = [d for d in bar_dir.iterdir() if d.is_dir()]

    results = {}
    all_featured = []

    for symbol_dir in symbol_dirs:
        asset = symbol_dir.name
        parquet_files = sorted(symbol_dir.glob("*.parquet"))
        if not parquet_files:
            results[asset] = {"status": "skipped", "reason": "no bar files"}
            continue

        try:
            bars = pd.read_parquet(parquet_files[0])
            if len(bars) < 30:
                results[asset] = {"status": "skipped", "reason": f"too few bars ({len(bars)})"}
                continue

            featured = calculator.calculate(bars, asset=asset)
            featured["asset"] = asset
            all_featured.append(featured)
            results[asset] = {"status": "completed", "rows": len(featured), "features": len(featured.columns)}
        except Exception as error:
            results[asset] = {"status": "failed", "error": str(error)}

    total_rows = sum(len(df) for df in all_featured) if all_featured else 0

    return {
        "total_assets": len(results),
        "completed": sum(1 for r in results.values() if r["status"] == "completed"),
        "total_rows": total_rows,
        "details": results,
    }


@feature_router.post("/calculate", summary="Submit feature calculation job")
async def submit_feature_calculation(request: FeatureCalculateRequest):
    """Submit feature calculation to the task manager."""
    try:
        task_manager = get_task_manager()
        task_payload = request.model_dump()

        task_id = task_manager.submit(
            name="feature_calculation",
            func=_run_feature_calculation,
            kwargs={"payload": task_payload},
        )

        return {
            "task_id": task_id,
            "state": "SUBMITTED",
            "payload": task_payload,
        }
    except Exception as error:
        logger.error(f"Failed to submit feature calculation: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@feature_router.get("/status/{task_id}", summary="Query feature calculation status")
async def get_feature_status(task_id: str):
    """Return execution status for a feature calculation task."""
    try:
        task_manager = get_task_manager()
        task_info = task_manager.get_task(task_id)
        if task_info is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task_info
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to query: {str(error)}")
