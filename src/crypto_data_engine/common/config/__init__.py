"""
Configuration module for crypto-data-engine.

Provides centralized configuration management with support for:
- Environment variables
- YAML configuration files
- Default values
"""
from crypto_data_engine.common.config.downloader_config import (
    BaseDownloadConfig,
    DataTypeConfig,
    ExchangeConfig,
    MultiExchangeDownloadConfig,
)
from crypto_data_engine.common.config.paths import (
    PROJECT_ROOT,
    DATA_ROOT,
    CONFIG_ROOT,
)
from crypto_data_engine.common.config.yaml_config import (
    get_download_config,
    load_yaml,
)
from crypto_data_engine.common.config.factor_config import (
    BUILTIN_FACTORS,
    AnalysisConfig,
    FactorConfig,
    FactorType,
    RollingMethod,
)

__all__ = [
    # Settings
    "Settings",
    "settings",
    "create_all_templates",
    # Config classes
    "BaseDownloadConfig",
    "DataTypeConfig",
    "ExchangeConfig",
    "MultiExchangeDownloadConfig",
    # YAML config
    "get_download_config",
    "load_yaml",
    # Paths
    "PROJECT_ROOT",
    "DATA_ROOT",
    "CONFIG_ROOT",
    # Factor config
    "BUILTIN_FACTORS",
    "AnalysisConfig",
    "FactorConfig",
    "FactorType",
    "RollingMethod",
]
