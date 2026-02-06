"""
Task execution configuration.

Lightweight task management without Celery dependency.
Supports both in-memory and Redis-based task state storage.
"""
from __future__ import annotations

from typing import Literal
from pydantic_settings import BaseSettings
from crypto_data_engine.common.config.paths import PROJECT_ROOT


class TaskConfig(BaseSettings):
    """Task execution and concurrency configuration."""
    
    # ============================================================================
    # Concurrency Settings
    # ============================================================================
    max_io_threads: int = 16
    """Maximum threads for I/O-bound operations (data loading, file reading)."""
    
    max_compute_processes: int = 8
    """Maximum processes for CPU-bound operations (bar aggregation, feature calculation)."""
    
    use_numba: bool = True
    """Enable Numba JIT compilation for performance-critical functions."""
    
    # ============================================================================
    # Task Storage Settings
    # ============================================================================
    task_store: Literal["memory", "redis"] = "redis"
    """Task state storage backend. 'memory' for development, 'redis' for production."""
    
    redis_url: str = "redis://localhost:6379/0"
    """Redis connection URL for task state storage."""
    
    task_ttl_seconds: int = 86400
    """Time-to-live for task state in seconds (default: 24 hours)."""
    
    # ============================================================================
    # Timeout Settings
    # ============================================================================
    backtest_timeout_seconds: int = 3600
    """Maximum time allowed for a single backtest run (default: 1 hour)."""
    
    data_load_timeout_seconds: int = 300
    """Maximum time for data loading operations (default: 5 minutes)."""
    
    aggregation_timeout_seconds: int = 600
    """Maximum time for bar aggregation operations (default: 10 minutes)."""
    
    # ============================================================================
    # Progress Tracking
    # ============================================================================
    enable_progress_tracking: bool = True
    """Enable real-time progress updates for long-running tasks."""
    
    progress_update_interval_seconds: float = 1.0
    """Interval between progress updates."""
    
    # ============================================================================
    # Resource Limits
    # ============================================================================
    max_memory_mb: int = 8192
    """Maximum memory usage in MB before triggering cleanup."""
    
    chunk_size: int = 100000
    """Default chunk size for streaming large datasets."""
    
    class Config:
        env_prefix = "TASK_"
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
