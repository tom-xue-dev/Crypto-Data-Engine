"""
Data download API routes.

Uses TaskManager (Redis-backed) for async task dispatch.
Supports:
- Triggering download jobs via API
- Real-time progress tracking via Redis
- Job status and history queries
"""
from datetime import datetime
from typing import List, Optional

import ccxt
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from crypto_data_engine.api.schemas.common import BaseResponse
from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.common.task_manager import get_task_manager, TaskStatus

download_router = APIRouter(prefix="/api/v1/download", tags=["Data Download"])

logger = get_logger(__name__)


# =============================================================================
# Request / Response schemas
# =============================================================================

class DownloadJobRequest(BaseModel):
    """Request to start a download pipeline job."""
    exchange: str = Field("binance_futures", description="Exchange name")
    symbols: Optional[List[str]] = Field(None, description="Symbols (None = all)")
    start_date: str = Field(..., description="Start date YYYY-MM")
    end_date: str = Field(..., description="End date YYYY-MM")
    max_threads: int = Field(8, ge=1, le=64, description="Download threads")


# =============================================================================
# Background job executor
# =============================================================================

def _run_download_pipeline_job(config: dict) -> dict:
    """Execute the full download pipeline in a background thread.

    Called by TaskManager.submit_io_task — receives a flat config dict.
    """
    from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download

    return run_download(
        exchange_name=config["exchange_name"],
        symbols=config.get("symbols"),
        start_date=config["start_date"],
        end_date=config["end_date"],
        max_threads=config.get("max_threads", 8),
        task_id=config.get("task_id"),
        task_manager=config.get("task_manager"),
    )


# =============================================================================
# API endpoints
# =============================================================================

@download_router.get("/exchanges", summary="List supported data sources")
async def get_exchanges():
    """Return all supported exchange data sources."""
    from crypto_data_engine.common.config.config_settings import settings

    sources = settings.downloader_cfg.list_all_exchanges()
    exchange_info = {
        name: {
            "base_url": cfg.base_url,
            "data_dir": str(cfg.data_dir),
            "supports_checksum": cfg.supports_checksum,
        }
        for name, cfg in sources.items()
    }
    return BaseResponse(data=exchange_info)


@download_router.get("/{source}/symbols", summary="List trading pairs for an exchange")
async def get_exchange_symbols(
    source: str = Path(..., description="Exchange name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of symbols"),
):
    """Return available trading pairs for the given exchange."""
    try:
        exchange_class = getattr(ccxt, source.lower())
        exchange = exchange_class()
        markets = exchange.load_markets()
        symbols = list(markets.keys())[:limit]
        return BaseResponse(data={
            "exchange": source,
            "symbols": symbols,
            "total": len(symbols),
        })
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Unsupported exchange: {source}")
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to get symbols: {str(error)}"
        )


@download_router.post("/jobs", summary="Start a download pipeline job")
async def create_download_job(request: DownloadJobRequest):
    """Create and start a download pipeline job.

    The job runs asynchronously in the background. Use the returned task_id
    to query progress via ``GET /jobs/{task_id}/progress``.
    """
    try:
        task_manager = get_task_manager()
        task_state = task_manager.create_task(metadata={
            "type": "download_pipeline",
            "exchange": request.exchange,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "symbols": request.symbols,
        })

        download_config = {
            "exchange_name": request.exchange,
            "symbols": request.symbols,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "max_threads": request.max_threads,
            "task_id": task_state.task_id,
            "task_manager": task_manager,
        }

        task_manager.submit_io_task(
            task_state.task_id,
            _run_download_pipeline_job,
            download_config,
        )

        return BaseResponse(data={
            "task_id": task_state.task_id,
            "status": "submitted",
            "exchange": request.exchange,
            "start_date": request.start_date,
            "end_date": request.end_date,
        })

    except Exception as error:
        logger.error(f"Failed to create download job: {error}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create download job: {str(error)}"
        )


@download_router.get("/jobs/{task_id}/progress", summary="Get live pipeline progress")
async def get_job_progress(task_id: str = Path(..., description="Task ID")):
    """Query real-time download pipeline progress from Redis.

    Returns detailed counters: total, downloaded, converted, failed, etc.
    """
    try:
        import redis as redis_lib
        from crypto_data_engine.common.config.config_settings import settings
        from crypto_data_engine.services.tick_data_scraper.downloader.downloader import (
            PROGRESS_KEY_PREFIX,
        )

        redis_client = redis_lib.from_url(
            settings.task_cfg.redis_url, decode_responses=True,
        )
        progress_key = f"{PROGRESS_KEY_PREFIX}:{task_id}"
        progress = redis_client.hgetall(progress_key)

        if not progress:
            raise HTTPException(status_code=404, detail=f"No progress found for task {task_id}")

        # Cast numeric fields
        for numeric_field in [
            "total_download", "total_convert",
            "downloaded", "download_failed", "download_skipped",
            "converted", "convert_failed",
        ]:
            if numeric_field in progress:
                progress[numeric_field] = int(progress[numeric_field])

        total_convert = progress.get("total_convert", 0)
        converted = progress.get("converted", 0)
        convert_failed = progress.get("convert_failed", 0)
        progress_ratio = (
            (converted + convert_failed) / total_convert
            if total_convert > 0 else 0.0
        )
        progress["progress_percent"] = round(progress_ratio * 100, 1)

        return BaseResponse(data=progress)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to query progress: {str(error)}"
        )


@download_router.get("/jobs/{task_id}/status", summary="Get task status from TaskManager")
async def get_job_status(task_id: str = Path(..., description="Task ID")):
    """Return TaskManager state for a download job (status, result, error)."""
    try:
        task_manager = get_task_manager()
        task_state = task_manager.get_task(task_id)
        if task_state is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return BaseResponse(data=task_state.to_dict())
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to query task status: {str(error)}"
        )


@download_router.get("/jobs", summary="List recent download jobs")
async def list_download_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
):
    """List recent download pipeline jobs."""
    try:
        task_manager = get_task_manager()
        filter_status = TaskStatus(status) if status else None
        tasks = task_manager.list_tasks(status=filter_status, limit=limit)

        # Filter to download_pipeline tasks only
        download_tasks = [
            task.to_dict()
            for task in tasks
            if task.metadata.get("type") == "download_pipeline"
        ]
        return BaseResponse(data={
            "jobs": download_tasks,
            "total": len(download_tasks),
        })
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to list jobs: {str(error)}"
        )
