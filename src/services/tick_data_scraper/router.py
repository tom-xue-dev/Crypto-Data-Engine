from fastapi import APIRouter, BackgroundTasks
from uuid import uuid4
from .pipeline import run_pipeline
from .worker import submit_task, get_status

router = APIRouter()

@router.post("/ingest", summary="触发 tick 下载")
async def ingest(symbol: str, start: str, end: str):
    task_id = submit_task(symbol, start, end)  # 放到进程池或 Ray
    return {"task_id": task_id}

@router.get("/status/{task_id}", summary="查询进度")
async def status(task_id: str):
    return get_status(task_id)
