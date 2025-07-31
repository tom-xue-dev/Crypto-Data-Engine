"""
Ray-powered two-stage pipeline
stage A: download zipped CSV  (IO bound)
stage B: unzip + convert to Parquet (CPU bound)
"""
import asyncio
import io, zipfile, pyarrow.parquet as pq, httpx, ray
import pyarrow.csv as pv
from datetime import datetime, timedelta
from typing import Dict



# ---------- stage A : IO ----------
@ray.remote(num_cpus=0.1, resources={"io": 1})
def fetch_gz(meta: dict) -> bytes:
    async def _inner():
        url = (
            f"{scraper_cfg.binance_url}/data/spot/daily/klines/"
            f"{meta['symbol']}/{meta['interval']}/"
            f"{meta['symbol']}-{meta['interval']}-{meta['date']}.zip"
        )
        async with httpx.AsyncClient(timeout=scraper_cfg.http_timeout) as cli:
            r = await cli.get(url)
            r.raise_for_status()
            return r.content

    return asyncio.run(_inner())

# ---------- stage B : CPU ----------
@ray.remote(num_cpus=1)
def gz_to_parquet(gz_bytes: bytes, meta: Dict) -> str:
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

# ---------- orchestrator ----------
@ray.remote
def download_pipeline(cfg: Dict, task_id: str) -> list[str]:
    """
    cfg = {symbol, start, end, interval, io_limit}
    1) 生成 meta 列表
    2) I/O 任务 fetch_gz.remote
    3) CPU 任务 gz_to_parquet.remote
    4) 返回写好的 Parquet 路径列表
    """
    start, end = map(lambda x: datetime.strptime(x, "%Y-%m-%d"), (cfg["start"], cfg["end"]))
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
    metas = [{"symbol": cfg["symbol"], "date": d, "interval": cfg["interval"]} for d in dates]

    # 1) 触发 IO
    io_refs  = [fetch_gz.options(resources={"io":1}).remote(m) for m in metas[: cfg["io_limit"]]]

    # 2) 动态补货 + 提交 CPU 任务
    parquet_refs = []
    remaining = metas[cfg["io_limit"]:]
    while io_refs:
        done, io_refs = ray.wait(io_refs, num_returns=1)
        gz_bytes = done[0]
        meta = metas[len(parquet_refs)]
        parquet_refs.append(gz_to_parquet.remote(gz_bytes, meta))
        if remaining:
            io_refs.append(fetch_gz.remote(remaining.pop(0)))

    return ray.get(parquet_refs)            # 列表[str]
