"""
Tick data download worker.

Provides the entry point for downloading exchange data.
Supports CLI invocation and API-triggered execution with TaskManager.
"""
from typing import List, Optional
from pathlib import Path

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)


def run_download(
    exchange_name: str = "binance_futures",
    symbols: Optional[List[str]] = None,
    start_date: str = "2020-01",
    end_date: str = "2020-01",
    data_dir: Optional[str] = None,
    max_threads: int = 8,
    task_id: Optional[str] = None,
    task_manager=None,
) -> dict:
    """
    Download tick data for the given exchange and date range.

    Handles the full pipeline: download → extract → convert to Parquet
    using Redis-backed queue communication.

    Args:
        exchange_name: Exchange adapter name (e.g. "binance", "binance_futures").
        symbols: List of symbols to download. None = download all available.
        start_date: Start date in "YYYY-MM" format.
        end_date: End date in "YYYY-MM" format.
        data_dir: Optional output directory override.
        max_threads: Number of concurrent download threads.
        task_id: Optional TaskManager task ID for API progress tracking.
        task_manager: Optional TaskManager instance for API integration.

    Returns:
        Pipeline summary dict with download/convert counts.
    """
    from crypto_data_engine.common.config.config_settings import settings
    from crypto_data_engine.services.tick_data_scraper.downloader.downloader import (
        DownloadContext,
        FileDownloader,
    )

    logger.info(f"Starting {exchange_name.upper()} data download")
    logger.info(f"Date range: {start_date} to {end_date}")

    config = settings.downloader_cfg.get_merged_config(exchange_name)
    if data_dir:
        config["data_dir"] = Path(data_dir)
    config["max_threads"] = max_threads
    redis_url = settings.task_cfg.redis_url
    logger.info(f"Data root: {config['data_dir']}")
    logger.info(f"Threads: {max_threads}")

    try:
        context = DownloadContext(config, start_date, end_date, symbols)
        downloader = FileDownloader(context, redis_url=redis_url)
        downloader.run_download_pipeline(
            task_id=task_id,
            task_manager=task_manager,
        )
        logger.info(f"File location: {config['data_dir']}")
        logger.info(f"{exchange_name.upper()} data download completed!")

        # Return progress summary if available
        job_id = task_id or f"dl_{exchange_name}_{int(__import__('time').time())}"
        progress = downloader.get_pipeline_progress(job_id)
        return progress or {"status": "completed"}
    except Exception as error:
        logger.error(f"Download failed: {error}")
        raise
