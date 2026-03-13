"""
Refactored downloader configuration system.
- Separate base configuration and exchange-specific configuration
- Support multiple exchanges
- Preserve flexibility and maintainability
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic_settings import BaseSettings
from crypto_data_engine.common.config.paths import PROJECT_ROOT,DATA_ROOT
from pydantic import BaseModel, Field, model_validator

class BaseDownloadConfig(BaseSettings):
    """Base downloader configuration – shared settings."""
    # --- Concurrency control ---
    max_threads: int = 16
    convert_processes: int = 4
    queue_size: int = 100
    # --- Network settings ---
    http_timeout: float = 60.0
    rate_limit_per_min: int = 1200
    # --- Retry strategy ---
    max_retries: int = 3
    base_retry_delay: float = 1.0
    exponential_backoff: bool = True
    # --- Data processing ---
    output_format: str = "parquet"  # parquet | csv | json
    compression: str = "brotli"     # brotli | gzip | snappy
    sort_by_timestamp: bool = True
    remove_duplicates: bool = True
    # --- Monitoring ---
    log_level: str = "INFO"
    enable_progress_bar: bool = True
    class Config:
        env_prefix = "DOWNLOADER_"
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class DataTypeConfig(BaseModel):
    """Configuration for a specific data type (aggTrades, bookTicker, etc.)."""
    base_url: Optional[str] = None  # Override exchange base_url if set
    file_name_format: str = "{symbol}-aggTrades-{year}-{month:02d}.zip"
    checksum_url_format: Optional[str] = None
    sub_dir: Optional[str] = None  # Subdirectory under data_dir for this data type


class ExchangeConfig(BaseModel):
    """Single exchange configuration."""
    name: str
    base_url: str
    symbol_info_url: str
    data_dir: Optional[Path] = Field(default=None)
    supports_checksum: bool = True
    file_name_format: str = "{symbol}-aggTrades-{year}-{month:02d}.zip"
    checksum_url_format: Optional[str] = None
    data_types: Dict[str, DataTypeConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_data_dir(cls, values):
        if values.data_dir is None:
            values.data_dir = DATA_ROOT / values.name
        return values

    def get_data_type_config(self, data_type: str) -> DataTypeConfig:
        """Get config for a specific data type, falling back to exchange defaults."""
        if data_type in self.data_types:
            return self.data_types[data_type]
        # Fall back: construct from exchange-level defaults
        return DataTypeConfig(
            file_name_format=self.file_name_format,
            checksum_url_format=self.checksum_url_format,
        )


class MultiExchangeDownloadConfig(BaseDownloadConfig):
    """Multiple exchange download configuration."""
    active_exchanges: List[str] = ["binance", "binance_futures"]
    exchange_configs: Dict[str, ExchangeConfig] = {
        "binance": ExchangeConfig(
            name="binance",
            base_url="https://data.binance.vision/data/spot/monthly/aggTrades",
            symbol_info_url="https://api.binance.com/api/v3/exchangeInfo",
            supports_checksum=True,
            checksum_url_format="https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}-aggTrades-{year}-{month:02d}.zip.CHECKSUM",
        ),
        "binance_futures": ExchangeConfig(
            name="binance_futures",
            base_url="https://data.binance.vision/data/futures/um/monthly/aggTrades",
            symbol_info_url="https://fapi.binance.com/fapi/v1/exchangeInfo",
            data_dir=Path("E:/data/binance_futures"),
            supports_checksum=True,
            file_name_format="{symbol}-aggTrades-{year}-{month:02d}.zip",
            checksum_url_format="https://data.binance.vision/data/futures/um/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip.CHECKSUM",
            data_types={
                "aggTrades": DataTypeConfig(
                    file_name_format="{symbol}-aggTrades-{year}-{month:02d}.zip",
                    checksum_url_format="https://data.binance.vision/data/futures/um/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip.CHECKSUM",
                ),
                "bookTicker": DataTypeConfig(
                    base_url="https://data.binance.vision/data/futures/um/monthly/bookTicker",
                    file_name_format="{symbol}-bookTicker-{year}-{month:02d}.zip",
                    checksum_url_format="https://data.binance.vision/data/futures/um/monthly/bookTicker/{symbol}/{symbol}-bookTicker-{year}-{month:02d}.zip.CHECKSUM",
                    sub_dir="bookTicker",
                ),
            },
        ),
        "binance_futures_bookticker": ExchangeConfig(
            name="binance_futures_bookticker",
            base_url="https://data.binance.vision/data/futures/um/monthly/bookTicker",
            symbol_info_url="https://fapi.binance.com/fapi/v1/exchangeInfo",
            data_dir=Path("E:/data/binance_futures/bookTicker"),
            supports_checksum=True,
            file_name_format="{symbol}-bookTicker-{year}-{month:02d}.zip",
            checksum_url_format="https://data.binance.vision/data/futures/um/monthly/bookTicker/{symbol}/{symbol}-bookTicker-{year}-{month:02d}.zip.CHECKSUM",
        ),
        "okx_futures": ExchangeConfig(
            name="okx_futures",
            base_url="https://www.okx.com/cdn/okex/traderecords/trades/monthly",
            symbol_info_url="https://www.okx.com/api/v5/public/instruments?instType=SWAP",
            data_dir=Path("E:/data"),
            supports_checksum=False,
            file_name_format="{symbol}-trades-{year}-{month:02d}.zip",
            checksum_url_format=None,
        ),
        "bybit": ExchangeConfig(
            name="bybit",
            base_url="https://public.bybit.com/trading",
            symbol_info_url="https://api.bybit.com/v5/market/instruments-info?category=spot",
            supports_checksum=False,
            file_name_format="{symbol}_{year}_{month:02d}.zip",
        ),
    }
    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """Return configuration for a specific exchange."""
        if exchange_name not in self.exchange_configs:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        return self.exchange_configs[exchange_name]

    def get_merged_config(self, exchange_name: str) -> Dict:
        """Merge base settings with exchange-specific configuration."""
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
        """Return all exchange configurations."""
        exchanges = {}
        for name, cfg in self.exchange_configs.items():
            exchanges[name] = cfg
        return exchanges


if __name__ == "__main__":
    # Configuration smoke test
    from config_settings import settings
    download_config = settings.downloader_cfg
    print("📦 Supported exchanges:")
    for name in download_config.exchange_configs.keys():
        print(f"  - {name}")

    print(f"\n📊 Binance configuration:")
    binance_config = download_config.get_exchange_config("binance")
    print(binance_config.data_dir)
    # print(f"  Data directory: {binance_config['data_dir']}")
    # print(f"  Download URL: {binance_config['base_url']}")
    #
    # print(f"\n📊 OKX configuration:")
    # okx_config = download_config.get_exchange_config("okx")
    # print(f"  Data directory: {okx_config['data_dir']}")
    # print(f"  Download URL: {okx_config['base_url']}")
    pass