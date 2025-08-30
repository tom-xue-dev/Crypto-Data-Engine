"""
重构的下载器配置系统
- 基础配置 + 交易所特定配置的分离设计
- 支持多交易所扩展
- 保持配置的灵活性和可维护性
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic_settings import BaseSettings
from crypto_data_engine.common.config.paths import PROJECT_ROOT,DATA_ROOT
from pydantic import BaseModel, Field, model_validator

class BaseDownloadConfig(BaseSettings):
    """下载器基础配置 - 通用设置"""
    # --- 并发控制 ---
    max_threads: int = 16
    convert_processes: int = 4
    queue_size: int = 100
    # --- 网络设置 ---
    http_timeout: float = 60.0
    rate_limit_per_min: int = 1200
    # --- 重试策略 ---
    max_retries: int = 3
    base_retry_delay: float = 1.0
    exponential_backoff: bool = True
    # --- 数据处理 ---
    output_format: str = "parquet"  # parquet | csv | json
    compression: str = "brotli"     # brotli | gzip | snappy
    sort_by_timestamp: bool = True
    remove_duplicates: bool = True
    # --- 监控 ---
    log_level: str = "INFO"
    enable_progress_bar: bool = True
    class Config:
        env_prefix = "DOWNLOADER_"
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class ExchangeConfig(BaseModel):
    """单个交易所配置"""
    name: str
    base_url: str
    symbol_info_url: str
    data_dir: Optional[Path] = Field(default=None)
    supports_checksum: bool = True
    file_name_format: str = "{symbol}-aggTrades-{year}-{month:02d}.zip"
    checksum_url_format: Optional[str] = None

    @model_validator(mode="after")
    def set_data_dir(cls, values):
        if values.data_dir is None:
            values.data_dir = DATA_ROOT / values.name
        return values


class MultiExchangeDownloadConfig(BaseDownloadConfig):
    """多交易所下载配置"""
    active_exchanges: List[str] = ["binance"]
    exchange_configs: Dict[str, ExchangeConfig] = {
        "binance": ExchangeConfig(
            name="binance",
            base_url="https://data.binance.vision/data/spot/monthly/aggTrades",
            symbol_info_url="https://api.binance.com/api/v3/exchangeInfo",
            supports_checksum=True,
            checksum_url_format="https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}-aggTrades-{year}-{month:02d}.zip.CHECKSUM"
        ),
        "okx": ExchangeConfig(
            name="okx",
            base_url="https://static.okx.com/cdn/okex/traderecords",
            symbol_info_url="https://www.okx.com/api/v5/public/instruments?instType=SPOT",
            supports_checksum=False,
            file_name_format="{symbol}_{year}{month:02d}.zip"
        ),
        "bybit": ExchangeConfig(
            name="bybit",
            base_url="https://public.bybit.com/trading",
            symbol_info_url="https://api.bybit.com/v5/market/instruments-info?category=spot",
            supports_checksum=False,
            file_name_format="{symbol}_{year}_{month:02d}.zip"
        )
    }
    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """获取指定交易所配置"""
        if exchange_name not in self.exchange_configs:
            raise ValueError(f"不支持的交易所: {exchange_name}")
        return self.exchange_configs[exchange_name]

    def get_merged_config(self, exchange_name: str) -> Dict:
        """获取合并后的配置（基础配置 + 交易所配置）"""
        exchange_config = self.get_exchange_config(exchange_name)

        return {
            "max_threads": self.max_threads,
            "convert_processes": self.convert_processes,
            "queue_size": self.queue_size,
            "http_timeout": self.http_timeout,
            "rate_limit_per_min": self.rate_limit_per_min,
            "max_retries": self.max_retries,
            "base_retry_delay": self.base_retry_delay,
            "exponential_backoff": self.exponential_backoff,
            "output_format": self.output_format,
            "compression": self.compression,
            "sort_by_timestamp": self.sort_by_timestamp,
            "remove_duplicates": self.remove_duplicates,
            "log_level": self.log_level,
            "enable_progress_bar": self.enable_progress_bar,
            "exchange_name": exchange_config.name,
            "base_url": exchange_config.base_url,
            "symbol_info_url": exchange_config.symbol_info_url,
            "data_dir": exchange_config.data_dir,
            "supports_checksum": exchange_config.supports_checksum,
            "file_name_format": exchange_config.file_name_format,
            "checksum_url_format": exchange_config.checksum_url_format,
        }

    def list_all_exchanges(self):
        """打印所有交易所配置"""
        exchanges = {}
        for name, cfg in self.exchange_configs.items():
            exchanges[name] = cfg
        return exchanges


if __name__ == "__main__":
    # 测试配置
    from config_settings import settings
    download_config = settings.downloader_cfg
    print("📦 支持的交易所:")
    for name in download_config.exchange_configs.keys():
        print(f"  - {name}")

    print(f"\n📊 Binance 配置:")
    binance_config = download_config.get_exchange_config("binance")
    print(binance_config.get_data_dir())
    # print(f"  数据目录: {binance_config['data_dir']}")
    # print(f"  下载URL: {binance_config['base_url']}")
    #
    # print(f"\n📊 OKX 配置:")
    # okx_config = download_config.get_exchange_config("okx")
    # print(f"  数据目录: {okx_config['data_dir']}")
    # print(f"  下载URL: {okx_config['base_url']}")
    pass