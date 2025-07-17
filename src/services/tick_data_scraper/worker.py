import ray, uuid
from pipeline import run_pipeline

ray.init(address="auto")                 # 本机 8 CPU；集群时换到 head IP

_tasks = {}                               # task_id -> ObjectRef

def submit_task(symbol, start, end):
    cfg = {"symbol": symbol, "start": start, "end": end, ...}
    task_id = uuid.uuid4().hex[:8]
    _tasks[task_id] = run_pipeline.remote(cfg)   # 返回一个 ObjectRef
    return task_id

def get_status(task_id):
    if task_id not in _tasks:
        return {"state": "not_found"}

    obj_ref = _tasks[task_id]
    ready, _ = ray.wait([obj_ref], timeout=0)
    if ready:
        try:
            output = ray.get(obj_ref)
            return {"state": "finished", "output": output}
        except Exception as e:
            return {"state": "failed", "error": str(e)}
    else:
        return {"state": "running"}
