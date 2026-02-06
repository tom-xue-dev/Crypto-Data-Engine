"""
Central settings loader.
Priority:   ① Environment variables from .evn file ② YAML  (e.g.: data_scraper_config.yaml)
            ③ Hard-coded fallback.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from crypto_data_engine.common.config.config_utils import create_template, LazyLoadConfig
from crypto_data_engine.common.config.paths import CONFIG_ROOT, PROJECT_ROOT
from crypto_data_engine.common.config.downloader_config import MultiExchangeDownloadConfig
from crypto_data_engine.common.config.aggregation_config import AggregationConfig
from crypto_data_engine.common.config.task_config import TaskConfig

# ------------------define ur common here ------------------

class BasicSettings(BaseSettings):
    class Config:
        env_file = PROJECT_ROOT/".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class TickDownloadConfig(BasicSettings):
    DataRoot: Path = PROJECT_ROOT/"data"/"tick_data"
    url: str = "https://data.binance.vision/data/spot/monthly/aggTrades"
    symbol_url:str = "https://api.binance.com/api/v3/exchangeInfo"
    io_limit: int = 8
    http_timeout: float = 60.0

    class Config:
        env_prefix = "TICK_"
        frozen = True

class ServerConfig(BasicSettings):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    reload: bool = False
    workers: int = 4

    class Config:
        env_prefix = "Server_"
        frozen = True

class Settings:
    server_cfg = LazyLoadConfig(ServerConfig)
    task_cfg: TaskConfig = LazyLoadConfig(TaskConfig)
    tick_download_cfg: TickDownloadConfig = LazyLoadConfig(TickDownloadConfig)
    downloader_cfg: MultiExchangeDownloadConfig = LazyLoadConfig(MultiExchangeDownloadConfig)
    aggregator_cfg: AggregationConfig = LazyLoadConfig(AggregationConfig)

def create_all_templates() -> None:
    """Instantiate all configuration classes and generate YAML templates."""
    # Register all configuration classes that require template generation here
    instances = [
        TickDownloadConfig(),
        TaskConfig(),
        ServerConfig(),
        MultiExchangeDownloadConfig(),
        AggregationConfig(),
    ]
    for inst in instances:
        create_template(inst, CONFIG_ROOT)


settings = Settings()

if __name__ == "__main__":
    pass

