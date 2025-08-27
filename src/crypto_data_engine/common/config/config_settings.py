"""
Central settings loader.
Priority:   ① Environment variables from .evn file ② YAML  (e.g.: data_scraper_config.yaml)
            ③ Hard-coded fallback.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from crypto_data_engine.common.config.config_utils import create_template, LazyLoadConfig
from crypto_data_engine.common.config.paths import CONFIG_ROOT, PROJECT_ROOT
from .downloader_config import MultiExchangeDownloadConfig

# ------------------define ur common here ------------------

class BasicSettings(BaseSettings):
    class Config:
        env_file = PROJECT_ROOT/".env"
        env_file_encoding = "utf-8"
        extra = "allow"

class CeleryConfig(BasicSettings):
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"

    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = ["json"]
    task_default_queue: str = "cpu"
    worker_max_tasks_per_child: int = 200
    task_acks_late: bool = True

    class Config:
        env_prefix = "CELERY_"
        extra = "ignore"
        # 🔥 添加字段别名映射
        field_aliases = {
            "result_backend": "CELERY_RESULT_BACKEND"
        }


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
    host: str = "localhost"
    port: int = 8080
    log_level: str = "INFO"
    reload: bool = False
    workers: int = 4

    class Config:
        env_prefix = "Server_"
        frozen = True

class DbConfig(BasicSettings):
    db_url: str = "postgresql+psycopg://admin:123456@localhost:5432/quantdb"
    db_pool_size: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    db_echo: bool = False

    class Config:
        env_prefix = "DB_"
        field_aliases = {
            "db_url": "DB_URL",
            "db_pool_size": "DB_POOL_SIZE",
            "db_pool_timeout": "DB_POOL_TIMEOUT",
            "db_pool_recycle": "DB_POOL_RECYCLE",
            "db_echo": "DB_ECHO"
        }


class Settings:
    server_cfg = LazyLoadConfig(ServerConfig)
    celery_cfg:CeleryConfig = LazyLoadConfig(CeleryConfig)
    tick_download_cfg:TickDownloadConfig = LazyLoadConfig(TickDownloadConfig)
    downloader_cfg:MultiExchangeDownloadConfig = LazyLoadConfig(MultiExchangeDownloadConfig)
    db_cfg = LazyLoadConfig(DbConfig)

def create_all_templates() -> None:
    """自动实例化所有配置类并生成对应 YAML 模板"""
    # 👉 在这里添加你需要注册的配置类
    instances = [
        TickDownloadConfig(),
        CeleryConfig(),
        ServerConfig(),
        MultiExchangeDownloadConfig(),
        DbConfig()
    ]
    for inst in instances:
        create_template(inst, CONFIG_ROOT)


settings = Settings()

if __name__ == "__main__":
    pass

