"""
Data download API routes.

Uses TaskManager (Redis-backed) for async task dispatch.
All state is managed via TaskManager — no database dependency.
"""
from datetime import datetime

import ccxt
from fastapi import APIRouter, HTTPException, Path, Query

from crypto_data_engine.api.schemas.common import (
    BaseResponse,
    ResponseCode,
)
from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.common.task_manager import get_task_manager

download_router = APIRouter(prefix="/api/v1/download", tags=["Data Download"])

logger = get_logger(__name__)


def _run_download_job(config: dict) -> dict:
    """Execute download in a worker thread/process."""
    from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download

    run_download(
        exchange_name=config["exchange_name"],
        symbols=config.get("symbols"),
        start_date=config["start_date"],
        end_date=config["end_date"],
        max_threads=config.get("max_threads", 8),
    )
    return {"status": "completed"}


@download_router.get("/exchanges", summary="List supported data sources")
async def get_sources():
    """Return all supported data sources."""
    from crypto_data_engine.common.config.config_settings import settings

    source = settings.downloader_cfg.list_all_exchanges()
    return BaseResponse(data=source)


@download_router.get("/{source}/symbols", summary="List trading pairs for an exchange")
async def get_source_symbols(
    source: str = Path(..., description="Exchange name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of symbols"),
):
    """Return available trading pairs for the given exchange."""
    try:
        exchange_class = getattr(ccxt, source.lower())
        exchange = exchange_class()
        markets = exchange.load_markets()
        symbols = list(markets.keys())[:limit]
        return {
            "exchange": source,
            "symbols": symbols,
            "total": len(symbols),
        }
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Unsupported exchange: {source}")
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to get symbols: {str(error)}"
        )


@download_router.post("/downloads/jobs", summary="Create download job")
async def create_download_job(request: "BatchDownloadRequest"):
    """Create a batch download job using TaskManager (no database)."""
    from crypto_data_engine.api.schemas.download import BatchDownloadRequest  # noqa: F811

    try:
        task_manager = get_task_manager()
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        submitted_tasks = []

        for month in request.months:
            for symbol in request.symbols:
                task_name = f"download_{request.exchange}_{symbol}_{request.year}_{month}"
                download_config = {
                    "exchange_name": request.exchange,
                    "symbols": [symbol],
                    "start_date": f"{request.year}-{month:02d}",
                    "end_date": f"{request.year}-{month:02d}",
                    "max_threads": 4,
                }
                task_id = task_manager.submit(
                    name=task_name,
                    func=_run_download_job,
                    kwargs={"config": download_config},
                )
                submitted_tasks.append({
                    "task_id": task_id,
                    "task_name": task_name,
                    "exchange": request.exchange,
                    "symbol": symbol,
                    "year": request.year,
                    "month": month,
                    "state": "SUBMITTED",
                })

        return {
            "job_id": job_id,
            "total_submitted": len(submitted_tasks),
            "tasks": submitted_tasks,
        }
    except Exception as error:
        logger.error(f"Failed to dispatch download tasks: {str(error)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create download job: {str(error)}"
        )


@download_router.get("/status/{task_id}", summary="Query download task status")
async def get_download_task_status(task_id: str):
    """Return execution status for a download task."""
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
