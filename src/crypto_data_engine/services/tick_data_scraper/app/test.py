# app.py - Download Microservice using FastAPI and Ray for distributed processing
import os
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import ray
from datetime import datetime
from dateutil.relativedelta import relativedelta
import asyncio
import io
import zipfile
import hashlib
import pandas as pd
import httpx

# Initialize Ray (assuming Ray is running or will be initialized here)
ray.init(ignore_reinit_error=True)


# Configuration (could be loaded from Config.py, but hardcoded defaults for simplicity)
class ScraperConfig:
    base_url = "https://data.binance.vision/data/spot/monthly/aggTrades"
    data_dir = "./data"
    http_timeout = 30
    completed_tasks_file = "completed_tasks.txt"


scraper_cfg = ScraperConfig()

# Global task tracking
_tasks: Dict[str, ray.ObjectRef] = {}
_meta: Dict[str, dict] = {}  # {progress, total, stage}


# Pydantic models for API
class DownloadConfig(BaseModel):
    symbols: List[str]  # e.g., ['NEOUSDT', 'LTCUSDT']
    start_date: str  # "YYYY-MM"
    end_date: str  # "YYYY-MM"
    max_io: int = 8  # Max concurrent IO tasks
    max_cpu: int = 4  # Max concurrent CPU tasks (for processing)


class TaskStatus(BaseModel):
    state: str
    progress: float = 0.0
    output: List[str] = None
    error: str = None


app = FastAPI(title="Download Microservice")


# ---------- Ray Remote Functions ----------

# ---------- API Endpoints ----------

@app.post("/submit_download", response_model=str)
async def submit_download(cfg: DownloadConfig, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex[:8]
    obj_ref = download_pipeline.remote(cfg, task_id)
    _tasks[task_id] = obj_ref
    return task_id


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    ready, _ = ray.wait([_tasks[task_id]], timeout=0)
    if ready:
        try:
            result: List[str] = ray.get(_tasks[task_id])
            return TaskStatus(state="finished", progress=1.0, output=result)
        except Exception as e:
            return TaskStatus(state="failed", progress=_meta.get(task_id, {}).get("progress", 0.0), error=str(e))
    else:
        meta = _meta.get(task_id, {"progress": 0.0})
        return TaskStatus(state="running", progress=meta["progress"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)