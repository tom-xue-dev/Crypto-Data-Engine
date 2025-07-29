import uuid, ray
from typing import Dict, List
from .pipeline import download_pipeline



_tasks: Dict[str, ray.ObjectRef] = {}
_meta: Dict[str, dict] = {}                # {progress, total}

# ---------- 提交 ----------
def submit_download(cfg: dict) -> str:
    task_id = uuid.uuid4().hex[:8]
    obj_ref = download_pipeline.remote(cfg, task_id)
    _tasks[task_id] = obj_ref
    _meta[task_id] = {"stage": "download", "progress": 0.0}
    return task_id

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
