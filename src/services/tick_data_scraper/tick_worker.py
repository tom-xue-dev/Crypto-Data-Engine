import uuid, ray
from pathlib import Path
from typing import Dict, List
from  datetime import datetime

from common.config.config_settings import Settings

_tasks: Dict[str, ray.ObjectRef] = {}
_meta: Dict[str, dict] = {}                # {progress, total}

# ---------- 提交 ----------
import httpx
import asyncio
import os
from pathlib import Path

async def submit_download(cfg: dict) -> dict:
    download_setting = Settings.tick_download_setting
    symbol = cfg["symbol"]
    year = int(cfg["year"])
    month = int(cfg["month"])
    base_url = download_setting.url

    date_str = f"{year}-{month:02d}"
    file_name = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{base_url}/{symbol}/{file_name}"
    out_path = Path(download_setting.DataRoot) / file_name
    parquet_path = out_path.with_suffix(".parquet")

    if parquet_path.exists():
        return {"status": "SKIPPED", "reason": "already exists"}

    async with httpx.AsyncClient(timeout=download_setting.http_timeout) as client:
        head_resp = await client.head(url)
        if head_resp.status_code != 200:
            return {"status": "SKIPPED", "reason": f"404 {url}"}
        try:
            get_resp = await client.get(url)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                async for chunk in get_resp.aiter_bytes(chunk_size=1024):
                    f.write(chunk)
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    return {
        "status": "SUCCESS",
        "symbol": symbol,
        "year": year,
        "month": month,
        "file_path": str(out_path),
    }


# ---------- 查询 ----------
def get_status(task_id: str) -> dict:
    if task_id not in _tasks:
        return {"state": "not_found"}

    ready, _ = ray.wait([_tasks[task_id]], timeout=0)
    if ready:
        try:
            result: List[str] = ray.get(_tasks[task_id])
            return {"state": "finished", "output": result}
        except Exception as e:
            return {"state": "failed", "error": str(e)}
    else:
        # 进度可选：使用 Ray 调度事件或 Redis 更新 _meta[task_id]["progress"]
        return {"state": "running", **_meta[task_id]}
