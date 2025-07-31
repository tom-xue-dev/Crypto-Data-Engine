import uuid, ray
from typing import Dict, List
from .pipeline import download_pipeline



_tasks: Dict[str, ray.ObjectRef] = {}
_meta: Dict[str, dict] = {}                # {progress, total}

# ---------- 提交 ----------
def submit_download(cfg: dict) -> dict:
    symbol = cfg["symbol"]
    start = datetime.fromisoformat(cfg["start_time"])
    end = datetime.fromisoformat(cfg["end_time"])

    # ✅ 模拟数据下载
    rows = [
        {"timestamp": start.isoformat(), "price": 48000.0, "volume": 1.2},
        {"timestamp": (start.replace(hour=1)).isoformat(), "price": 48100.0, "volume": 2.5},
        {"timestamp": (start.replace(hour=2)).isoformat(), "price": 48250.0, "volume": 0.9},
    ]

    # ✅ 保存路径
    out_dir = Path(f"data/{symbol.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol}_{start.date()}_{end.date()}.csv"
    file_path = out_dir / filename

    # ✅ 写入 CSV 文件
    with open(file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "price", "volume"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✓] 下载完成: {file_path}")
    return {
        "symbol": symbol,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "file_path": str(file_path)
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
