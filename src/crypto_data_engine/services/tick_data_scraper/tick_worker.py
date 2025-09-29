from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

def run_download(
        exchange_name: str = "binance",
        symbols: list = None,
        start_date: str = "2020-01",
        end_date: str = "2020-01",
        max_threads: int = 8
):
    """
    Simplified single-exchange download helper – downloads only, no post-processing.
    """

    logger.info(f"\n🚀 Start downloading {exchange_name.upper()} data")
    logger.info(f"📅 Date range: {start_date} to {end_date}")
    from crypto_data_engine.common.config.config_settings import settings
    config = settings.downloader_cfg.get_merged_config(exchange_name)
    logger.info(f"📂 data root: {config['data_dir']}")
    logger.info(f"🎯 threads: {config['max_threads']}")
    from crypto_data_engine.services.tick_data_scraper.downloader.downloader import DownloadContext, FileDownloader
    try:
        # Create download context and downloader
        context = DownloadContext(config,start_date,end_date,symbols)
        downloader = FileDownloader(context)
        downloader.run_download_pipeline(config)
        # Show download results
        logger.info(f"📂 File location: {config['data_dir']}")
        logger.info(f"\n🎉 {exchange_name.upper()} data download completed!")
    except Exception as e:
        logger.warning(f"❌ Download failed: {e}")
        raise

