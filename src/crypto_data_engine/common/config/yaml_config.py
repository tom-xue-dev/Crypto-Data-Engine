"""
YAML configuration loader.

Loads configuration from YAML files with support for:
- Default config templates generation
- Merging multiple config sources (defaults → YAML file → env vars)
- Type-safe access via dict-like interface
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import logging
import yaml

from crypto_data_engine.common.config.paths import CONFIG_ROOT, PROJECT_ROOT

# Use stdlib logging to avoid circular import with loguru-based logger
_logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        _logger.warning(f"Config file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base dict (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Default YAML templates
# ---------------------------------------------------------------------------

DOWNLOAD_CONFIG_TEMPLATE = """\
# =============================================================================
# Crypto Data Engine - Download Configuration
# =============================================================================

# Global download settings
download:
  max_threads: 16
  convert_processes: 4
  queue_size: 100
  http_timeout: 60.0
  rate_limit_per_min: 1200
  max_retries: 3
  base_retry_delay: 1.0
  exponential_backoff: true
  output_format: parquet    # parquet | csv | json
  compression: brotli       # brotli | gzip | snappy
  sort_by_timestamp: true
  remove_duplicates: true
  log_level: INFO
  enable_progress_bar: true

# Exchange-specific configurations
exchanges:
  binance:
    base_url: https://data.binance.vision/data/spot/monthly/aggTrades
    symbol_info_url: https://api.binance.com/api/v3/exchangeInfo
    supports_checksum: true
    data_types:
      aggTrades:
        file_name_format: "{symbol}-aggTrades-{year}-{month:02d}.zip"
        checksum_url_format: "https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip.CHECKSUM"

  binance_futures:
    base_url: https://data.binance.vision/data/futures/um/monthly/aggTrades
    symbol_info_url: https://fapi.binance.com/fapi/v1/exchangeInfo
    data_dir: E:/data/binance_futures
    supports_checksum: true
    data_types:
      aggTrades:
        file_name_format: "{symbol}-aggTrades-{year}-{month:02d}.zip"
        checksum_url_format: "https://data.binance.vision/data/futures/um/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip.CHECKSUM"
      bookTicker:
        base_url: https://data.binance.vision/data/futures/um/monthly/bookTicker
        file_name_format: "{symbol}-bookTicker-{year}-{month:02d}.zip"
        checksum_url_format: "https://data.binance.vision/data/futures/um/monthly/bookTicker/{symbol}/{symbol}-bookTicker-{year}-{month:02d}.zip.CHECKSUM"

  okx_futures:
    base_url: https://www.okx.com/cdn/okex/traderecords/trades/monthly
    symbol_info_url: https://www.okx.com/api/v5/public/instruments?instType=SWAP
    data_dir: E:/data
    supports_checksum: false
    data_types:
      aggTrades:
        file_name_format: "{symbol}-trades-{year}-{month:02d}.zip"

  bybit:
    base_url: https://public.bybit.com/trading
    symbol_info_url: https://api.bybit.com/v5/market/instruments-info?category=spot
    supports_checksum: false
    data_types:
      aggTrades:
        file_name_format: "{symbol}_{year}_{month:02d}.zip"

# Data storage paths
paths:
  data_root: E:/data
  futures_data_root: E:/data

# Redis configuration (for download pipeline)
redis:
  url: redis://localhost:6379/0
"""


def create_download_config_template(output_dir: Optional[Path] = None) -> Path:
    """Create the default download config YAML template."""
    target_dir = output_dir or CONFIG_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "download_config.yaml"
    target.write_text(DOWNLOAD_CONFIG_TEMPLATE, encoding="utf-8")
    _logger.info(f"Created download config template: {target}")
    return target


def get_download_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load download configuration from YAML.

    Search order:
    1. Explicit config_path argument
    2. PROJECT_ROOT / config.yaml
    3. CONFIG_ROOT / download_config.yaml
    4. Built-in defaults
    """
    # Try explicit path
    if config_path and config_path.exists():
        return load_yaml(config_path)

    # Try project root
    project_config = PROJECT_ROOT / "config.yaml"
    if project_config.exists():
        return load_yaml(project_config)

    # Try config templates dir
    template_config = CONFIG_ROOT / "download_config.yaml"
    if template_config.exists():
        return load_yaml(template_config)

    # Return defaults parsed from template
    return yaml.safe_load(DOWNLOAD_CONFIG_TEMPLATE)
