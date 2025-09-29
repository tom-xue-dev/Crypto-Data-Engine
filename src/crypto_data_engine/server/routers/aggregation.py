from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.common.config.config_settings import settings
from crypto_data_engine.server.constants.request_schema import AggregateRequest
from task_manager.celery_app import celery_app

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/aggregate", tags=["Bar Aggregation"])





@router.post("/bars", summary="Submit bar aggregation job (dispatch to Celery)")
def submit_aggregate_job(req: AggregateRequest) -> Dict[str, Any]:
    try:
        # Use defaults if user omitted params
        merged = settings.aggregator_cfg.resolve_defaults(req.bar_type)
        bar_type = merged["bar_type"]
        threshold = req.threshold or merged.get("threshold")

        task_payload = {
            "exchange": req.exchange,
            "symbols": req.symbols,  # If None, worker will derive from DB
            "bar_type": bar_type,
            "threshold": threshold,
        }

        result = celery_app.send_task(
            "bar.aggregate",
            kwargs=task_payload,
            queue="cpu",
        )
        return {
            "celery_task_id": result.id,
            "state": result.state,
            "payload": task_payload,
        }
    except Exception as e:
        logger.error(f"Failed to submit aggregation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/celery/status/{celery_task_id}", summary="Query Celery task status for aggregation job")
async def get_aggregate_celery_status(celery_task_id: str):
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


@router.get("/tasks", summary="List aggregation tasks from DB")
async def list_aggregate_tasks(
    exchange: Optional[str] = None,
    symbol: Optional[str] = None,
    bar_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    try:
        from crypto_data_engine.db.repository.aggregate import AggregateTaskRepository
        from crypto_data_engine.db.constants import TaskStatus

        filters: Dict[str, Any] = {}
        if exchange:
            filters['exchange'] = exchange
        if symbol:
            filters['symbol'] = symbol
        if bar_type:
            filters['bar_type'] = bar_type
        if status:
            # best-effort: map to enum; if invalid, ignore
            try:
                filters['status'] = TaskStatus(status)
            except Exception:
                pass

        records = AggregateTaskRepository.get_all(limit=limit, offset=offset, order_by="part_date", desc_order=True, **filters)
        data = [
            {
                'id': r.id,
                'exchange': r.exchange,
                'symbol': r.symbol,
                'bar_type': r.bar_type,
                'part_date': str(r.part_date) if r.part_date else None,
                'status': r.status.value if getattr(r, 'status', None) else None,
                'file_name': getattr(r, 'file_name', None),
                'file_path': getattr(r, 'file_path', None),
                'task_start': str(r.task_start) if getattr(r, 'task_start', None) else None,
                'task_end': str(r.task_end) if getattr(r, 'task_end', None) else None,
            }
            for r in records
        ]
        return {
            'count': len(data),
            'items': data,
        }
    except Exception as e:
        logger.error(f"Failed to list aggregate tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

