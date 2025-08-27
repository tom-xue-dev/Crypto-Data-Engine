"""
Celery-powered two-stage pipeline
stage A: download zipped CSV  (IO bound)    -> Celery task: fetch_gz
stage B: unzip + convert to Parquet (CPU bound) -> Celery task: gz_to_parquet
Orchestrator builds chain(fetch_gz -> gz_to_parquet) per date, groups them, and uses chord to aggregate results.
"""
import io
import zipfile
import pyarrow.csv as pv
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import group, chain, chord

from crypto_data_engine.common.config.paths import DATA_ROOT
from crypto_data_engine.services.tick_data_scraper.app.test import scraper_cfg
from task_manager.celery_app import celery_app

# Optional: route IO/CPU tasks to dedicated queues
DOWNLOAD_QUEUE = "download_tasks"
CPU_QUEUE = "bar_tasks"

# Python
# 下载任务：requests + 重试 + 流式下载 + 幂等
import os, tempfile, shutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _session(timeout: tuple[float, float] = (5.0, 30.0)) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        respect_retry_after_header=True,
    )
    s.headers.update({"User-Agent": "tick-downloader/1.0"})
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=64)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.request_timeout = timeout
    return s

@celery_app.task(
    bind=True,
    name="tick.fetch_gz",
    autoretry_for=(requests.RequestException,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
    soft_time_limit=120,
    time_limit=150,
)
def fetch_gz(self, meta: dict) -> dict:
    """
    返回:
      {"status": "SUCCESS"|"SKIPPED"|"ERROR", "meta": meta, "path": str|None, "reason": str|None}
    路径是 zip 文件在共享卷/本地磁盘的位置
    """
    s = _session()
    url = (
        f"{meta['base_url']}/data/spot/daily/klines/"
        f"{meta['symbol']}/{meta['interval']}/"
        f"{meta['symbol']}-{meta['interval']}-{meta['date']}.zip"
    )
    out_dir = DATA_ROOT / meta["symbol"] / meta["interval"] / meta["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{meta['symbol']}-{meta['interval']}-{meta['date']}.zip"
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "SUCCESS", "meta": meta, "path": str(out_path), "reason": "HIT_LOCAL"}
    try:
        with s.head(url, allow_redirects=True, timeout=s.request_timeout) as h:
            if h.status_code == 404:
                return {"status": "SKIPPED", "reason": f"404 {url}", "meta": meta, "path": None}
            h.raise_for_status()

        # 临时文件写完再原子移动，避免并发读到半截文件
        with tempfile.NamedTemporaryFile(dir=out_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with s.get(url, stream=True, timeout=s.request_timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
        shutil.move(str(tmp_path), str(out_path))  # 原子替换

        return {"status": "SUCCESS", "meta": meta, "path": str(out_path), "reason": None}

    except SoftTimeLimitExceeded:
        # 清理半截文件
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        finally:
            return {"status": "ERROR", "meta": meta, "path": None, "reason": "soft time limit exceeded"}


@celery_app.task(
    bind=True,
    name="tick.gz_to_parquet",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def gz_to_parquet(self, gz_bytes: bytes, meta: Dict) -> str:
    """
    解压 → 读 CSV → 写 Parquet，返回最终文件路径
    """
    zf = zipfile.ZipFile(io.BytesIO(gz_bytes))
    csv_name = zf.namelist()[0]
    df = pv.read_csv(zf.open(csv_name))
    out_dir = scraper_cfg.DataRoot / "tick" / meta["symbol"] / meta["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{csv_name[:-4]}.parquet"
    pq.write_table(df, out_file, compression="zstd")
    return str(out_file)


@celery_app.task(name="tick.aggregate")
def aggregate(results: List[str]) -> List[str]:
    """
    chord 回调：收集所有结果（Parquet 路径列表）
    """
    return results


@celery_app.task(bind=True, name="tick.download_pipeline")
def download_pipeline(self, cfg: Dict, task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    cfg = {symbol, start, end, interval, io_limit?}
    - 生成每个日期的 meta
    - 对每个 meta 创建 chain(fetch_gz -> gz_to_parquet)
    - 用 group 并行 + chord 聚合，返回 chord 的 result_id 以供查询
    """
    start, end = map(lambda x: datetime.strptime(x, "%Y-%m-%d"), (cfg["start"], cfg["end"]))
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
    metas = [{"symbol": cfg["symbol"], "date": d, "interval": cfg["interval"]} for d in dates]

    # 为每个 meta 构造 fetch -> convert 的链条，并路由到不同队列
    chains = [
        chain(
            fetch_gz.s(m).set(queue=DOWNLOAD_QUEUE),
            gz_to_parquet.s(m).set(queue=CPU_QUEUE),
        )
        for m in metas
    ]

    header = group(chains)
    res = chord(header)(aggregate.s())
    # 返回可用于查询最终结果的 id（AsyncResult(res.id).get() -> List[str]）
    return {"result_id": res.id, "total": len(chains), "task_id": task_id}
