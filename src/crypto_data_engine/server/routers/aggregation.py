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
            "bars.aggregate",
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

