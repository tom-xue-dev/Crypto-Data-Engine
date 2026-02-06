import logging
import re
from pathlib import Path
from typing import Type
from pydantic import BaseModel
import yaml
from crypto_data_engine.common.config.paths import CONFIG_ROOT

# Use stdlib logging here to avoid circular import with project logger
logger = logging.getLogger(__name__)


class LazyLoadConfig:
    def __init__(self, config_cls):
        self.config_cls = config_cls
        self._value = None

    def __get__(self, instance, owner):
        if self._value is None:
            self._value = load_config(self.config_cls)
        return self._value


def to_snake_case(name: str) -> str:
    """

    convert the class name into yaml file name
    """
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower() + ".yaml"

def load_config(cls: Type[BaseModel]) -> BaseModel:
    """Load YAML configuration for the given class and instantiate it."""
    filename = f"{to_snake_case(cls.__name__)}"
    path = CONFIG_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"❌ could not find file: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    expected_keys = cls.model_fields.keys()
    filtered_data = {k: v for k, v in data.items() if k in expected_keys}
    return cls(**filtered_data)

def create_template(instance: BaseModel, output_dir: Path = Path("")) -> None:
    """Create YAML template for the provided settings instance."""
    filename = to_snake_case(instance.__class__.__name__)
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(instance.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
    logger.info(f"Created settings template: {path.resolve()}")
