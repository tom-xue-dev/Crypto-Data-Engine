from services.tick_data_scraper.app.path_utils import get_sorted_assets, get_asset_file_path
from services.tick_data_scraper.app.pre_process import PreprocessContext
import ray
# ---------- 外层控制函数 ----------


def run_pipeline(config: dict) -> list[str]:
    """拆分任务 → Ray 并行 → 收集输出文件路径"""
    ctx = PreprocessContext(config)
    assets = get_sorted_assets(ctx.data_dir, ctx.suffix_filter)

    # 过滤 2025 数据
    asset_infos = [
        (asset, size, ctx) for asset, size in assets
        if not _skip_2025(asset, ctx)
    ]

    # —— 广播只读 context，到每个 worker 内共享内存，避免复制 ——
    ctx_id = ray.put(ctx)

    # —— 提交 Ray 任务 ——
    object_ids = [
        process_asset.remote(asset, size, ctx_id)
        for asset, size, _ in asset_infos
    ]

    # —— 收集结果，同时可以在这里做进度上报 ——
    done_paths = []
    for oid in ray.progress(object_ids):         # 需要 pip install ray[default]
        res = ray.get(oid)
        if res:
            print(f"{res} done!")
            done_paths.append(res)

    return done_paths


# ---------- Ray 远程任务 ----------
@ray.remote
def process_asset(asset: str, size: int, ctx_id) -> str | None:
    ctx: PreprocessContext = ray.get(ctx_id)  # 共享对象
    paths = get_asset_file_path(asset, data_dir=ctx.data_dir)
    if not paths:
        return None
    # 省略实际聚合 / 写 parquet 逻辑 ...
    output_path = f".../{asset}.parquet"
    return output_path


# ---------- 工具 ----------
def _skip_2025(asset: str, ctx) -> bool:
    paths = get_asset_file_path(asset, data_dir=ctx.data_dir)
    return not paths or paths[0][-15:-11] == "2025"
