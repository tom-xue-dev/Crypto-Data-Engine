"""
Bar aggregation API routes.

Uses TaskManager (Redis-backed) for async task dispatch.
All state is managed via TaskManager — no database dependency.
"""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from crypto_data_engine.api.schemas.download import AggregateRequest
from crypto_data_engine.common.config.config_settings import settings
from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.common.task_manager import get_task_manager

logger = get_logger(__name__)

aggregation_router = APIRouter(prefix="/api/v1/aggregate", tags=["Bar Aggregation"])


def _run_aggregation_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute bar aggregation in a worker process."""
    from pathlib import Path

    import pandas as pd

    from crypto_data_engine.services.bar_aggregator import aggregate_bars

    exchange = payload["exchange"]
    symbols = payload["symbols"]
    bar_type = payload["bar_type"]
    threshold = payload["threshold"]

    results = {}
    for symbol in symbols or []:
        data_dir = settings.downloader_cfg.get_exchange_config(exchange).data_dir
        parquet_path = Path(data_dir) / symbol
        if not parquet_path.exists():
            results[symbol] = {"status": "skipped", "reason": "no data"}
            continue

        parquet_files = sorted(parquet_path.glob("*.parquet"))
        if not parquet_files:
            results[symbol] = {"status": "skipped", "reason": "no parquet files"}
            continue

        all_ticks = pd.concat([pd.read_parquet(f) for f in parquet_files])
        bars = aggregate_bars(all_ticks, bar_type, threshold)
        results[symbol] = {
            "status": "completed",
            "bar_count": len(bars),
        }

    return results


@aggregation_router.post("/bars", summary="Submit bar aggregation job")
async def submit_aggregate_job(req: AggregateRequest) -> Dict[str, Any]:
    """Submit a bar aggregation task to the task manager."""
    try:
        merged = settings.aggregator_cfg.resolve_defaults(req.bar_type)
        bar_type = merged["bar_type"]
        threshold = req.threshold or merged.get("threshold")

        task_payload = {
            "exchange": req.exchange,
            "symbols": req.symbols,
            "bar_type": bar_type,
            "threshold": threshold,
        }

        task_manager = get_task_manager()
        task_id = task_manager.submit(
            name=f"aggregate_{req.exchange}_{bar_type}",
            func=_run_aggregation_job,
            kwargs={"payload": task_payload},
        )

        return {
            "task_id": task_id,
            "state": "SUBMITTED",
            "payload": task_payload,
        }
    except Exception as error:
        logger.error(f"Failed to submit aggregation job: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@aggregation_router.get("/status/{task_id}", summary="Query aggregation task status")
async def get_aggregate_status(task_id: str):
    """Return execution status for an aggregation task."""
    try:
        task_manager = get_task_manager()
        task_info = task_manager.get_task(task_id)
        if task_info is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task_info
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to query: {str(error)}"
        )
