"""
Pydantic schemas for API request/response models.
"""

from .backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestStatus,
    BacktestResult,
    TradeRecord,
    PerformanceMetrics,
)
from .common import (
    BaseResponse,
    JobResponse,
    MetricsResponse,
    ResponseCode,
    TaskResponse,
)
from .download import (
    AggregateRequest,
    BatchDownloadRequest,
    DownloadRequest,
)
from .strategy import (
    StrategyInfo,
    StrategyParam,
    StrategyListResponse,
    StrategyValidateRequest,
    StrategyValidateResponse,
)

__all__ = [
    # Backtest schemas
    "BacktestRequest",
    "BacktestResponse",
    "BacktestStatus",
    "BacktestResult",
    "TradeRecord",
    "PerformanceMetrics",
    # Common response schemas
    "BaseResponse",
    "ResponseCode",
    "TaskResponse",
    "JobResponse",
    "MetricsResponse",
    # Download / Aggregation schemas
    "DownloadRequest",
    "BatchDownloadRequest",
    "AggregateRequest",
    # Strategy schemas
    "StrategyInfo",
    "StrategyParam",
    "StrategyListResponse",
    "StrategyValidateRequest",
    "StrategyValidateResponse",
]
