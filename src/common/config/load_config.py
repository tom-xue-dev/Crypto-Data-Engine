from common.config.config_settings import DataScraperConfig, CeleryConfig, to_snake_case, ServerConfig, CONFIG_DIR, \
    RayConfig
from typing import Type
from pydantic import BaseModel
import yaml


def load_config(cls: Type[BaseModel]) -> BaseModel:
    """根据类名自动加载对应 YAML 并实例化配置对象"""
    filename = f"{to_snake_case(cls.__name__)}"
    path = CONFIG_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"❌ could not find file: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    expected_keys = cls.model_fields.keys()
    filtered_data = {k: v for k, v in data.items() if k in expected_keys}
    return cls(**filtered_data)

server_cfg = load_config(ServerConfig)
scraper_cfg = load_config(DataScraperConfig)
ray_cfg = load_config(RayConfig)
celery_cfg = load_config(CeleryConfig)
