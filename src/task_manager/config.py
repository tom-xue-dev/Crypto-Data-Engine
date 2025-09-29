from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict,Any


class TaskType(Enum):
    """define your tasks here"""
    TICK_DOWNLOAD = "tick_download"
    BAR_AGGREGATION = "bar_aggregation"
    FEATURE_GENERATION = "feature_generation"
    SIGNAL_GENERATION = "signal_generation"
    BACKTESTING = "backtesting"
    VISUALIZATION = "visualization"

@dataclass
class TaskConfig:
    """Task configuration."""
    task_type: TaskType
    name: str
    module_path: str
    function_name: str
    queue: str = "default"
    routing_key: Optional[str] = None
    time_limit: Optional[int] = None
    soft_time_limit: Optional[int] = None
    retry_policy: Optional[Dict[str, Any]] = None
