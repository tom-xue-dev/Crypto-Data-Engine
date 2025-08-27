import re
from pathlib import Path
from typing import Type
from pydantic import BaseModel
import yaml
from crypto_data_engine.common.config.paths import CONFIG_ROOT


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
    """根据类名自动加载对应 YAML 并实例化配置对象"""
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
    """create yaml file based on the common instance"""
    filename = to_snake_case(instance.__class__.__name__)
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(instance.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
    print(f"✅ create setting template : {path.resolve()}")
