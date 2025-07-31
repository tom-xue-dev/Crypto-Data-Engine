from uuid import uuid4
import requests
from fastapi import APIRouter, Body

from common.config.config_settings import settings
from common.constants.request_schema import TickDownloadRequest, CryptoSymbolRequest
from common.constants.response import BaseResponse
from common.constants.response_code import ResponseCode
from server.startup_server import logger
from task_manager.celery_app import celery_app



tick_router = APIRouter()

@tick_router.post("/download-tick")
def download_tick(payload: TickDownloadRequest):
    task_id = str(uuid4())
    celery_app.send_task(
        "tick.download",
        args=[payload.model_dump()],
        task_id=task_id
    )
    return {
        "task_id": task_id,
        "status": "PENDING"
    }

@tick_router.get("/symbols")
def get_symbols(request:CryptoSymbolRequest = Body(...)):
    url = settings.tick_download_setting.symbol_url
    logger.info(f"trying to get symbols from {url}...")
    response = requests.get(url)
    data = response.json()
    symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
    logger.info(f"got {len(symbols)} symbols first symbol is {symbols[0]}")
    if request:
        symbols = [s for s in symbols if s.endswith(request.suffix)]
    return BaseResponse(
        code=ResponseCode.SUCCESS,
        message="SUCCESS",
        data={"symbols": symbols}
    )
