
def run_simple_download(
        exchange_name: str = "binance",
        symbols: list = None,
        start_date: str = "2020-01",
        end_date: str = "2022-03",
        max_threads: int = 8
):
    """
    简化的单交易所数据下载函数 - 只下载，不处理
    """

    print(f"\n🚀 开始下载 {exchange_name.upper()} 数据")
    print(f"📅 时间范围: {start_date} 到 {end_date}")
    # 获取交易所配置
    from crypto_data_engine.common.config.config_settings import settings
    config = settings.downloader_cfg.get_merged_config(exchange_name)
    # 覆盖配置参数
    config.update({
        'start_date': start_date,
        'end_date': end_date,
        'symbols': symbols or "auto",
        'max_threads': max_threads,
    })
    # 如果需要特定的交易对过滤
    if symbols:
        config['symbols'] = symbols

    print(f"📂 数据目录: {config['data_dir']}")
    print(f"🎯 线程数: {config['max_threads']}")
    from crypto_data_engine.services.tick_data_scraper.downloader.downloader import DownloadContext, FileDownloader
    try:
        # 创建下载上下文和下载器
        context = DownloadContext(config)
        downloader = FileDownloader(context)

        # 运行纯下载流水线
        downloader.run_download_pipeline(config)

        # 显示下载结果
        downloaded_files = downloader.get_downloaded_files()
        print(f"\n📁 本次会话共有 {len(downloaded_files)} 个文件可用")
        print(f"📂 文件位置: {config['data_dir']}")

        print(f"\n🎉 {exchange_name.upper()} 数据下载完成！")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise


