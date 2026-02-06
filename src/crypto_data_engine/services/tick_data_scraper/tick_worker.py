"""
Tick data download worker.

Provides a simple entry point for downloading exchange data.
Supports both spot and futures markets.
"""
from typing import List, Optional

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)


def run_download(
    exchange_name: str = "binance_futures",
    symbols: Optional[List[str]] = None,
    start_date: str = "2020-01",
    end_date: str = "2020-01",
    max_threads: int = 8,
) -> None:
    """
    Download tick data for the given exchange and date range.

    Handles the full pipeline: download -> extract -> convert to Parquet.

    Args:
        exchange_name: Exchange adapter name (e.g. "binance", "binance_futures").
        symbols: List of symbols to download. None = download all available.
        start_date: Start date in "YYYY-MM" format.
        end_date: End date in "YYYY-MM" format.
        max_threads: Number of concurrent download threads.
    """
    from crypto_data_engine.common.config.config_settings import settings
    from crypto_data_engine.services.tick_data_scraper.downloader.downloader import (
        DownloadContext,
        FileDownloader,
    )

    logger.info(f"Starting {exchange_name.upper()} data download")
    logger.info(f"Date range: {start_date} to {end_date}")

    config = settings.downloader_cfg.get_merged_config(exchange_name)
    config["max_threads"] = max_threads
    logger.info(f"Data root: {config['data_dir']}")
    logger.info(f"Threads: {max_threads}")

    try:
        context = DownloadContext(config, start_date, end_date, symbols)
        downloader = FileDownloader(context)
        downloader.run_download_pipeline()
        logger.info(f"File location: {config['data_dir']}")
        logger.info(f"{exchange_name.upper()} data download completed!")
    except Exception as error:
        logger.error(f"Download failed: {error}")
        raise
