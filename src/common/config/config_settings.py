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

from common.config.paths import CONFIG_DIR, PROJECT_ROOT


# ------------------Define project base paths -----------------------



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



class DataScraperConfig(BasicSettings):
    DataRoot: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data_test")
    binance_url: str = "https://data.binance.vision"
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

# ------------------ helper function------------------

def to_snake_case(name: str) -> str:
    """

    convert the class name into yaml file name
    """
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower() + ".yaml"


def create_template(instance: BaseModel, output_dir: Path = Path("")) -> None:
    """create yaml file based on the common instance"""
    filename = to_snake_case(instance.__class__.__name__)
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(instance.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)

    print(f"✅ create setting template : {path.resolve()}")


def create_all_templates() -> None:
    """自动实例化所有配置类并生成对应 YAML 模板"""
    # 👉 在这里添加你需要注册的配置类
    instances = [
        DataScraperConfig(),
        CeleryConfig(),
        RayConfig(),
        ServerConfig(),
    ]
    for inst in instances:
        create_template(inst, CONFIG_DIR)



if __name__ == "__main__":
    import sys
    if "--init" in sys.argv:
        create_all_templates()

