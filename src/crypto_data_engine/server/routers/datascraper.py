"""
Data scraper API routes following the documented schema.
"""
from datetime import datetime, date
from fastapi import APIRouter, Body, HTTPException, Query, Path, BackgroundTasks
import ccxt
from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.db.repository.download import DownloadTaskRepository
from crypto_data_engine.db.models.download import TaskStatus
from crypto_data_engine.server.constants.request_schema import BatchDownloadRequest
from crypto_data_engine.server.constants.response import JobResponse, TaskResponse, BaseResponse
from crypto_data_engine.server.constants.response_code import ResponseCode
from task_manager.celery_app import celery_app

datascraper_router = APIRouter(prefix="/api/v1/download", tags=["Data Scraper"])

logger = get_logger(__name__)
# ==================== Source management ====================

@datascraper_router.get("/exchanges", summary="List supported data sources")
async def get_sources():
    """Return all supported data sources."""
    from crypto_data_engine.common.config.config_settings import settings
    source = settings.downloader_cfg.list_all_exchanges()
    response = BaseResponse(data=source)
    return response

@datascraper_router.get("/{source}/symbols", summary="List trading pairs for an exchange")
async def get_source_symbols(
        source: str = Path(..., description="Exchange name"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of symbols")
):
    try:
        # Use ccxt to fetch exchange information
        exchange_class = getattr(ccxt, source.lower())
        exchange = exchange_class()
        markets = exchange.load_markets()
        symbols = list(markets.keys())[:limit]
        return {
            "exchange": source,
            "symbols": symbols,
            "total": len(symbols)
        }
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Unsupported exchange: {source}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fail to get the symbol: {str(e)}")

# ==================== Download task management ====================


@datascraper_router.post("/downloads/jobs", response_model=JobResponse, summary="Create download job")
async def create_download_job(request: BatchDownloadRequest):
    """Create a batch download job and submit to Celery."""
    try:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        year_month_pairs = [(request.year, month) for month in request.months]
        created_tasks, skipped = DownloadTaskRepository.create_batch_tasks(
            exchange=request.exchange,
            symbols=request.symbols,
            year_month_pairs=year_month_pairs,
        )
        task_responses = []
        for task in created_tasks:
            task_response = TaskResponse(
                id=task.id,
                exchange=task.exchange,
                symbol=task.symbol,
                year=task.year,
                month=task.month,
                status=task.status.value,
                file_name=task.file_name,
                file_size=task.file_size,
                local_path=task.local_path,
                task_start=task.task_start,
                task_end=task.task_end
            )
            task_responses.append(task_response)
    except Exception as e:
        logger.error(f"create download tasks failed: {str(e)}")
        return BaseResponse(code=ResponseCode.DB_ERROR,message="DB_ERROR",data=e)

    try:
        # Submit tasks to Celery
        celery_task_ids = []
        for i, task in enumerate(created_tasks):
            celery_config = {
                'exchange_name': task.exchange,
                'symbols': [task.symbol],
                'start_date': f"{task.year}-{task.month:02d}",
                'end_date': f"{task.year}-{task.month:02d}",
                'max_threads': 4,
                'task_id': task.id
            }
            celery_result = celery_app.send_task(
                'tick.download',
                args=[celery_config],
                queue='io_intensive'
            )
            celery_task_ids.append({
                'db_task_id': task.id,
                'celery_task_id': celery_result.id,
                'status': celery_result.state
            })
            # Update task status
            DownloadTaskRepository.update_status(task.id, TaskStatus.PENDING)
        return JobResponse(
            job_id=job_id,
            created_tasks=task_responses,  # Use pre-built response objects
            skipped_tasks=skipped,
            total_created=len(created_tasks),
            total_skipped=len(skipped)
        )

    except Exception as e:
        logger.error(f"Failed to dispatch tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create download job: {str(e)}")


@datascraper_router.get("/celery/status/{celery_task_id}", summary="Query Celery task status")
async def get_celery_task_status(celery_task_id: str):
    """Return execution status for a Celery task."""
    try:
        from celery.result import AsyncResult

        result = AsyncResult(celery_task_id, app=celery_app)

        return {
            'celery_task_id': celery_task_id,
            'status': result.state,
            'result': result.result if result.ready() else None,
            'info': result.info,
            'successful': result.successful(),
            'failed': result.failed()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query: {str(e)}")

