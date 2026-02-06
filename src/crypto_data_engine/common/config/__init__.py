"""
Configuration module for crypto-data-engine.

Provides centralized configuration management with support for:
- Environment variables
- YAML configuration files
- Default values
"""
from crypto_data_engine.common.config.config_settings import (
    Settings,
    settings,
    create_all_templates,
    BasicSettings,
    ServerConfig,
    TickDownloadConfig,
)
from crypto_data_engine.common.config.task_config import TaskConfig
from crypto_data_engine.common.config.aggregation_config import AggregationConfig
from crypto_data_engine.common.config.downloader_config import (
    BaseDownloadConfig,
    ExchangeConfig,
    MultiExchangeDownloadConfig,
)
from crypto_data_engine.common.config.paths import (
    PROJECT_ROOT,
    DATA_ROOT,
    CONFIG_ROOT,
)

__all__ = [
    # Settings
    "Settings",
    "settings",
    "create_all_templates",
    # Config classes
    "BasicSettings",
    "ServerConfig",
    "TickDownloadConfig",
    "TaskConfig",
    "AggregationConfig",
    "BaseDownloadConfig",
    "ExchangeConfig",
    "MultiExchangeDownloadConfig",
    # Paths
    "PROJECT_ROOT",
    "DATA_ROOT",
    "CONFIG_ROOT",
]
