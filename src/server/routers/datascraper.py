from fastapi import APIRouter

from server.routers.request_schema import DLReq
from services.tick_data_scraper.worker import get_status

# 分发任务用的

data_scraper_router = APIRouter()

@data_scraper_router.post("/download")
async def trigger_download(req: DLReq):
    task_id = dispatch_tick_download.delay(req.model_dump())
    return {"task_id": task_id.id}

@data_scraper_router.get("/status/{task_id}")
async def status(task_id: str):
    return get_status(task_id)