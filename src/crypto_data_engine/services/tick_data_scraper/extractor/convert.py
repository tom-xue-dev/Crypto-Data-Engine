from pathlib import Path
import shutil
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def extract_archive(directory: str, file_name: str) -> dict:
    """
    Extract an archive located in the target directory.
    Supports zip/tar/tar.gz (handled by `shutil.unpack_archive`).
    Returns: {"archive": <archive path>, "out_dir": <extracted directory>, "files": [list of files]}
    """
    directory_p = Path(directory)
    archive = directory_p / file_name
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    out_dir_p = directory_p / archive.stem
    out_dir_p.mkdir(parents=True, exist_ok=True)
    # Extract files
    shutil.unpack_archive(str(archive), str(out_dir_p))  # Must be str
    # Collect extracted files
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
    Convert files matching `pattern` under the extracted directory to Parquet.
    Defaults to converting *.csv to *.parquet; override output_dir if needed.
    Returns a list of generated parquet file paths.
    """
    csv_read_kwargs = csv_read_kwargs or {}
    parquet_kwargs = parquet_kwargs or {
        "engine": "pyarrow",
        "compression": "zstd",
        "index": False,
    }

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
        # Target path (preserve relative layout)
        rel = src.relative_to(src_dir) if src.parent != src_dir else src.name
        out_path = out_root / Path(rel).with_suffix(".parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read/write operations
        df = pd.read_csv(src, **csv_read_kwargs)  # Use read_json for JSON sources
        df.to_parquet(out_path, **parquet_kwargs)  # e.g. engine="pyarrow", compression="zstd"

        parquet_paths.append(str(out_path))
        logger.info(f"Converted {src} -> {out_path}")

    logger.info(f"Parquet generated: {len(parquet_paths)} files")
    return parquet_paths
