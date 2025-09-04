from fastapi import APIRouter, HTTPException

from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.server.constants.request_schema import BarProcessorRequest
from crypto_data_engine.server.constants.response import BaseResponse

bar_aggr_router = APIRouter(prefix="/api/v1/bars", tags=["bar aggregator"])
logger = get_logger(__name__)

@bar_aggr_router.post("/aggregate", summary="Aggregate tick data into bars")
async def aggregate_bars(payload: BarProcessorRequest):
    """Trigger bar aggregation via Celery."""
    from task_manager.celery_app import celery_app
    try:
        result = celery_app.send_task(
            "bar.aggregate", kwargs=payload.model_dump()
        )
        return BaseResponse(data={"task_id": result.id})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))