"""
Centralized settings with YAML configuration support.

Loads configuration from YAML files and provides typed access
to all subsystem configs (downloader, task manager, paths, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from crypto_data_engine.common.config.downloader_config import MultiExchangeDownloadConfig
from crypto_data_engine.common.config.paths import PROJECT_ROOT, CONFIG_ROOT
from crypto_data_engine.common.config.yaml_config import (
    get_download_config,
    create_download_config_template,
)


class TaskConfig(BaseModel):
    """Task manager configuration."""
    task_store: str = "memory"  # memory | redis
    redis_url: str = "redis://localhost:6379/0"
    max_io_threads: int = 8
    max_compute_processes: int = 4


class Settings(BaseSettings):
    """Application-wide settings, backed by YAML + env vars."""

    downloader_cfg: MultiExchangeDownloadConfig = Field(
        default_factory=MultiExchangeDownloadConfig
    )
    task_cfg: TaskConfig = Field(default_factory=TaskConfig)

    class Config:
        env_prefix = "CDE_"
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """Create Settings instance enriched with YAML configuration."""
        yaml_cfg = get_download_config(config_path)
        instance = cls()

        # Apply YAML download settings to downloader_cfg
        download_section = yaml_cfg.get("download", {})
        if download_section:
            for key, value in download_section.items():
                if hasattr(instance.downloader_cfg, key):
                    setattr(instance.downloader_cfg, key, value)

        # Apply YAML redis/task settings
        redis_section = yaml_cfg.get("redis", {})
        if redis_section:
            if "url" in redis_section:
                instance.task_cfg.redis_url = redis_section["url"]

        # Apply YAML paths
        paths_section = yaml_cfg.get("paths", {})
        if paths_section:
            if "data_root" in paths_section:
                import crypto_data_engine.common.config.paths as paths_mod
                paths_mod.FUTURES_DATA_ROOT = Path(paths_section["data_root"])

        return instance


def create_all_templates():
    """Create all YAML config templates."""
    create_download_config_template()


# Singleton instance (lazy-loaded from YAML)
settings = Settings.from_yaml()
