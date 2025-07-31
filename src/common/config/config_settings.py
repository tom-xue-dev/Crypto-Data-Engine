"""
Central settings loader.
Priority:   ① Environment variables from .evn file ② YAML  (e.g.: data_scraper_config.yaml)
            ③ Hard-coded fallback.
"""
import os
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml
import re
from common.config.config_utils import load_config, to_snake_case, create_template, LazyLoadConfig
from common.config.paths import CONFIG_DIR, PROJECT_ROOT


# ------------------define ur common here ------------------

class BasicSettings(BaseSettings):
    class Config:
        env_file = PROJECT_ROOT/".env"
        env_file_encoding = "utf-8"
        extra = "allow"

class CeleryConfig(BasicSettings):
    broker_url: str = "redis://localhost:6379/0"
    backend_url: str = "redis://localhost:6379/1"

    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = ["json"]
    task_default_queue: str = "cpu"
    worker_max_tasks_per_child: int = 200
    task_acks_late: bool = True
    class Config:
        env_prefix = "CELERY_"
        extra = "ignore"



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
    host: str = "192.168.1.100"
    port: int = 8080
    log_level: str = "INFO"
    reload: bool = False
    workers: int = 4

    class Config:
        env_prefix = "Server_"
        frozen = True

class RayConfig(BasicSettings):
    pass

class Settings:
    server_cfg = LazyLoadConfig(ServerConfig)
    scraper_cfg = LazyLoadConfig(TickDownloadConfig)
    ray_cfg = LazyLoadConfig(RayConfig)
    celery_cfg = LazyLoadConfig(CeleryConfig)
    tick_download_setting = LazyLoadConfig(TickDownloadConfig)

def create_all_templates() -> None:
    """自动实例化所有配置类并生成对应 YAML 模板"""
    # 👉 在这里添加你需要注册的配置类
    instances = [
        TickDownloadConfig(),
        CeleryConfig(),
        RayConfig(),
        ServerConfig(),
    ]
    for inst in instances:
        create_template(inst, CONFIG_DIR)


settings = Settings()

if __name__ == "__main__":
    pass

