from pathlib import Path
import shutil
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def extract_archive(directory: str, file_name: str) -> dict:
    """
    解压指定目录中的压缩包。
    支持 zip/tar/tar.gz 等（由 shutil.unpack_archive 决定）。
    返回：{"archive": <压缩包路径>, "out_dir": <解压目录>, "files": [解压出的文件列表]}
    """
    directory_p = Path(directory)
    archive = directory_p / file_name
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    out_dir_p = directory_p / archive.stem
    out_dir_p.mkdir(parents=True, exist_ok=True)
    # 解压
    shutil.unpack_archive(str(archive), str(out_dir_p))  # 注意：需要是 str
    # 收集解压出的文件
    files = [str(p) for p in out_dir_p.rglob("*") if p.is_file()]
    logger.info(f"Extracted {archive} -> {out_dir_p}, {len(files)} files")
    return {"archive": str(archive), "out_dir": str(out_dir_p), "files": files}


def convert_dir_to_parquet(
    extracted_dir: str,
    pattern: str = "*.csv",
    output_dir: str | None = None,
    csv_read_kwargs: dict | None = None,
    parquet_kwargs: dict | None = None,
) -> list[str]:
    """
    将解压目录中匹配 pattern 的文件批量转为 Parquet。
    默认将 *.csv 转为同名 .parquet；可用 output_dir 覆盖输出目录。
    返回：生成的 parquet 文件路径列表
    """
    csv_read_kwargs = csv_read_kwargs or {}
    parquet_kwargs = parquet_kwargs or {}

    src_dir = Path(extracted_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"extracted_dir not found: {src_dir}")

    out_root = Path(output_dir) if output_dir else src_dir
    out_root.mkdir(parents=True, exist_ok=True)

    parquet_paths: list[str] = []
    src_files = sorted([p for p in src_dir.rglob(pattern) if p.is_file()])

    if not src_files:
        logger.warning(f"No files matched {pattern} under {src_dir}")
        return parquet_paths

    for src in src_files:
        # 目标路径（保持相对结构）
        rel = src.relative_to(src_dir) if src.parent != src_dir else src.name
        out_path = out_root / Path(rel).with_suffix(".parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 读写
        df = pd.read_csv(src, **csv_read_kwargs)  # 若是 JSON，可改成 read_json
        df.to_parquet(out_path, **parquet_kwargs)  # 例如 engine="pyarrow", compression="zstd"

        parquet_paths.append(str(out_path))
        logger.info(f"Converted {src} -> {out_path}")

    logger.info(f"Parquet generated: {len(parquet_paths)} files")
    return parquet_paths
