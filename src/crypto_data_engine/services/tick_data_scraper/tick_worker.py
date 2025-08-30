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
    简化的单交易所数据下载函数 - 只下载，不处理
    """

    logger.info(f"\n🚀 start download {exchange_name.upper()} data")
    logger.info(f"📅 时间范围: {start_date} 到 {end_date}")
    from crypto_data_engine.common.config.config_settings import settings
    config = settings.downloader_cfg.get_merged_config(exchange_name)
    logger.info(f"📂 data root: {config['data_dir']}")
    logger.info(f"🎯 threads: {config['max_threads']}")
    from crypto_data_engine.services.tick_data_scraper.downloader.downloader import DownloadContext, FileDownloader
    try:
        # 创建下载上下文和下载器
        context = DownloadContext(config,start_date,end_date,symbols)
        downloader = FileDownloader(context)
        downloader.run_download_pipeline(config)
        # 显示下载结果
        logger.info(f"📂 文件位置: {config['data_dir']}")
        logger.info(f"\n🎉 {exchange_name.upper()} 数据下载完成！")
    except Exception as e:
        logger.warning(f"❌ 下载失败: {e}")
        raise

