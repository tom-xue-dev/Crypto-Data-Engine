"""
Data inventory API routes.

Scan tick data directories to report what has been downloaded.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from crypto_data_engine.common.config.config_settings import settings
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

data_inventory_router = APIRouter(prefix="/api/v1/data", tags=["Data Inventory"])


def _scan_symbol_dir(symbol_dir: Path) -> Dict[str, Any]:
    """Scan a single symbol directory and return summary info."""
    parquet_files = sorted(symbol_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    dates = []
    for file in parquet_files:
        parts = file.stem.split("-")
        if len(parts) >= 4:
            dates.append(f"{parts[-2]}-{parts[-1]}")

    total_size_bytes = sum(f.stat().st_size for f in parquet_files)

    # Estimate row count from first file
    estimated_rows_per_file = 0
    try:
        sample = pd.read_parquet(parquet_files[0], columns=["price"])
        estimated_rows_per_file = len(sample)
    except Exception:
        pass

    return {
        "symbol": symbol_dir.name,
        "file_count": len(parquet_files),
        "date_range_start": min(dates) if dates else None,
        "date_range_end": max(dates) if dates else None,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "estimated_total_rows": estimated_rows_per_file * len(parquet_files),
    }


@data_inventory_router.get("/inventory", summary="List all downloaded tick data")
async def get_data_inventory(
    data_dir: Optional[str] = Query(None, description="Override tick data directory"),
):
    """Scan tick data directory and return per-symbol summary."""
    try:
        if data_dir:
            tick_root = Path(data_dir)
        else:
            exchange_config = settings.downloader_cfg.get_exchange_config("binance_futures")
            tick_root = Path(exchange_config.data_dir)

        if not tick_root.exists():
            return {"data_dir": str(tick_root), "symbols": [], "total_symbols": 0}

        symbol_dirs = sorted([d for d in tick_root.iterdir() if d.is_dir()])
        inventory = []
        for symbol_dir in symbol_dirs:
            info = _scan_symbol_dir(symbol_dir)
            if info:
                inventory.append(info)

        return {
            "data_dir": str(tick_root),
            "total_symbols": len(inventory),
            "symbols": inventory,
        }
    except Exception as error:
        logger.error(f"Failed to scan data inventory: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@data_inventory_router.get("/inventory/{symbol}", summary="Get detailed info for a symbol")
async def get_symbol_inventory(
    symbol: str,
    data_dir: Optional[str] = Query(None, description="Override tick data directory"),
):
    """Return detailed file list for a single symbol."""
    try:
        if data_dir:
            tick_root = Path(data_dir)
        else:
            exchange_config = settings.downloader_cfg.get_exchange_config("binance_futures")
            tick_root = Path(exchange_config.data_dir)

        symbol_dir = tick_root / symbol.upper()
        if not symbol_dir.exists():
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        parquet_files = sorted(symbol_dir.glob("*.parquet"))
        files = []
        for file in parquet_files:
            files.append({
                "name": file.name,
                "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
            })

        summary = _scan_symbol_dir(symbol_dir)
        return {
            "symbol": symbol.upper(),
            "summary": summary,
            "files": files,
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Failed to get symbol inventory: {error}")
        raise HTTPException(status_code=500, detail=str(error))
