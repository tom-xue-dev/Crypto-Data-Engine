"""
Bar aggregation API routes.

Supports batch aggregation with Redis-backed pipeline and progress tracking.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from crypto_data_engine.api.schemas.common import BaseResponse
from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.common.task_manager import get_task_manager, TaskStatus

aggregation_router = APIRouter(prefix="/api/v1/aggregate", tags=["Bar Aggregation"])

logger = get_logger(__name__)


# =============================================================================
# Request schemas
# =============================================================================

class BatchAggregationRequest(BaseModel):
    """Request to start a batch aggregation job."""

    tick_data_dir: str = Field("./data/tick_data", description="Tick data directory")
    output_dir: str = Field("./data/bars", description="Output directory")
    bar_type: str = Field("dollar_bar", description="Bar type (time_bar, tick_bar, volume_bar, dollar_bar)")
    threshold: str = Field("1000000", description="Threshold (5min for time_bar, 1000 for tick_bar, 1000000 for dollar/volume_bar)")
    symbols: Optional[List[str]] = Field(None, description="Symbols (None = all)")
    workers: int = Field(4, ge=1, le=16, description="Parallel workers")
    force: bool = Field(False, description="Re-aggregate existing bars")


# =============================================================================
# Background job executor
# =============================================================================

def _run_batch_aggregation_job(config: dict) -> dict:
    """Execute batch aggregation in a background thread."""
    from crypto_data_engine.services.bar_aggregator.batch_aggregator import BatchAggregator

    aggregator = BatchAggregator(
        tick_data_dir=config["tick_data_dir"],
        output_dir=config["output_dir"],
        bar_type=config["bar_type"],
        threshold=config["threshold"],
    )

    aggregator.run_aggregation_pipeline(
        symbols=config.get("symbols"),
        workers=config.get("workers", 4),
        force=config.get("force", False),
        task_id=config.get("task_id"),
        task_manager=config.get("task_manager"),
    )

    # Return progress summary
    job_id = config.get("task_id")
    if job_id:
        progress = aggregator.get_pipeline_progress(job_id)
        return progress or {"status": "completed"}
    return {"status": "completed"}


# =============================================================================
# API endpoints
# =============================================================================

@aggregation_router.post("/jobs", summary="Start batch aggregation job")
async def create_aggregation_job(request: BatchAggregationRequest):
    """Create and start a batch aggregation job.

    The job runs asynchronously in the background. Use the returned task_id
    to query progress via ``GET /jobs/{task_id}/progress``.
    """
    try:
        task_manager = get_task_manager()
        task_state = task_manager.create_task(
            metadata={
                "type": "batch_aggregation",
                "bar_type": request.bar_type,
                "threshold": request.threshold,
                "symbols": request.symbols,
            }
        )

        job_config = {
            "tick_data_dir": request.tick_data_dir,
            "output_dir": request.output_dir,
            "bar_type": request.bar_type,
            "threshold": request.threshold,
            "symbols": request.symbols,
            "workers": request.workers,
            "force": request.force,
            "task_id": task_state.task_id,
            "task_manager": task_manager,
        }

        task_manager.submit_io_task(
            task_state.task_id,
            _run_batch_aggregation_job,
            job_config,
        )

        return BaseResponse(
            data={
                "task_id": task_state.task_id,
                "status": "submitted",
                "bar_type": request.bar_type,
                "threshold": request.threshold,
            }
        )

    except Exception as error:
        logger.error(f"Failed to create aggregation job: {error}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create job: {str(error)}"
        )


@aggregation_router.get("/jobs/{task_id}/progress", summary="Get aggregation progress")
async def get_aggregation_progress(task_id: str = Path(..., description="Task ID")):
    """Query real-time aggregation pipeline progress from Redis."""
    try:
        import redis as redis_lib
        from crypto_data_engine.common.config.config_settings import settings
        from crypto_data_engine.services.bar_aggregator.batch_aggregator import (
            AGGREGATE_PROGRESS_KEY,
        )

        redis_client = redis_lib.from_url(
            settings.task_cfg.redis_url, decode_responses=True
        )
        progress_key = f"{AGGREGATE_PROGRESS_KEY}:{task_id}"
        progress = redis_client.hgetall(progress_key)

        if not progress:
            raise HTTPException(
                status_code=404, detail=f"No progress found for task {task_id}"
            )

        # Cast numeric fields
        for numeric_field in ["total_tasks", "completed", "failed"]:
            if numeric_field in progress:
                progress[numeric_field] = int(progress[numeric_field])

        total = progress.get("total_tasks", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        progress_ratio = (completed + failed) / total if total > 0 else 0.0
        progress["progress_percent"] = round(progress_ratio * 100, 1)

        return BaseResponse(data=progress)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to query progress: {str(error)}"
        )


@aggregation_router.get("/jobs/{task_id}/status", summary="Get task status")
async def get_aggregation_status(task_id: str = Path(..., description="Task ID")):
    """Return TaskManager state for an aggregation job."""
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


@aggregation_router.get("/jobs", summary="List recent aggregation jobs")
async def list_aggregation_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
):
    """List recent batch aggregation jobs."""
    try:
        task_manager = get_task_manager()
        filter_status = TaskStatus(status) if status else None
        tasks = task_manager.list_tasks(status=filter_status, limit=limit)

        # Filter to batch_aggregation tasks only
        aggregation_tasks = [
            task.to_dict()
            for task in tasks
            if task.metadata.get("type") == "batch_aggregation"
        ]
        return BaseResponse(
            data={
                "jobs": aggregation_tasks,
                "total": len(aggregation_tasks),
            }
        )
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to list jobs: {str(error)}"
        )
