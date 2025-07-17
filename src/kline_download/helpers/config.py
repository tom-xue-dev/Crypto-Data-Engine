import os

import yaml

_config_path = os.path.join("config", "config.yml")


class _ConfigMeta(type):
    _config = None

    def initialize(cls):
        """初始化配置文件"""
        if cls._config is None:
            with open(_config_path, "r", encoding="utf-8") as file:
                cls._config = yaml.safe_load(file)

    def __call__(cls, config_name):
        cls.initialize()
        return cls._config.get(config_name)


class Config(metaclass=_ConfigMeta):
    pass
