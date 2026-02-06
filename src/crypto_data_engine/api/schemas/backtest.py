"""
Pydantic schemas for backtest API.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestModeEnum(str, Enum):
    """Backtest execution modes."""
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"
    MULTI_ASSET = "multi_asset"


class TaskStatus(str, Enum):
    """Backtest task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskConfigSchema(BaseModel):
    """Risk management configuration."""
    max_position_size: float = Field(default=0.1, ge=0, le=1)
    max_total_exposure: float = Field(default=1.0, ge=0)
    max_leverage: float = Field(default=3.0, ge=1)
    max_drawdown: float = Field(default=0.2, ge=0, le=1)
    daily_loss_limit: float = Field(default=0.05, ge=0, le=1)
    stop_loss_strategies: List[str] = Field(default_factory=list)
    stop_loss_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class CostConfigSchema(BaseModel):
    """Cost model configuration."""
    commission_rate: float = Field(default=0.001, ge=0)
    maker_rate: float = Field(default=0.0002, ge=0)
    taker_rate: float = Field(default=0.0005, ge=0)
    slippage_rate: float = Field(default=0.0005, ge=0)
    funding_enabled: bool = True
    leverage_enabled: bool = True
    default_funding_rate: float = Field(default=0.0001)
    default_leverage_rate: float = Field(default=0.00005)


class AssetPoolConfigSchema(BaseModel):
    """Asset pool configuration."""
    method: str = Field(default="turnover")
    top_k: int = Field(default=100, ge=1)
    lookback_days: int = Field(default=30, ge=1)
    rebalance_frequency: str = Field(default="M")
    min_price: Optional[float] = None
    min_volume: Optional[float] = None
    custom_assets: Optional[List[str]] = None


class StrategyConfigSchema(BaseModel):
    """Strategy configuration."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class BacktestRequest(BaseModel):
    """Request model for running a backtest."""
    # Strategy
    strategy: StrategyConfigSchema

    # Execution mode
    mode: BacktestModeEnum = BacktestModeEnum.CROSS_SECTIONAL

    # Capital
    initial_capital: float = Field(default=1_000_000.0, gt=0)

    # Time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Data settings
    bar_type: str = Field(default="time")
    price_col: str = Field(default="close")
    time_col: str = Field(default="start_time")

    # Rebalancing (cross-sectional mode)
    rebalance_frequency: str = Field(default="W-MON")

    # Assets
    assets: Optional[List[str]] = None
    use_asset_pool: bool = True
    asset_pool_config: Optional[AssetPoolConfigSchema] = None

    # Risk management
    risk_config: Optional[RiskConfigSchema] = None

    # Cost model
    cost_config: Optional[CostConfigSchema] = None

    # Data source
    data_dir: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": {
                    "name": "momentum",
                    "params": {
                        "lookback_period": 20,
                        "long_count": 10,
                        "short_count": 10
                    }
                },
                "mode": "cross_sectional",
                "initial_capital": 1000000,
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-12-31T23:59:59",
                "rebalance_frequency": "W-MON"
            }
        }


class BacktestResponse(BaseModel):
    """Response model for backtest submission."""
    task_id: str
    status: TaskStatus
    message: str
    created_at: datetime


class BacktestStatus(BaseModel):
    """Status of a backtest task."""
    task_id: str
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0, le=1)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class TradeRecord(BaseModel):
    """Single trade record."""
    timestamp: datetime
    asset: str
    direction: str  # "BUY" or "SELL"
    quantity: float
    price: float
    value: float
    commission: float
    slippage: float
    pnl: Optional[float] = None


class PerformanceMetrics(BaseModel):
    """Backtest performance metrics."""
    # Returns
    total_return: float
    annualized_return: float

    # Risk
    volatility: float
    sharpe_ratio: float
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    calmar_ratio: Optional[float] = None

    # Trading
    total_trades: int
    win_rate: float
    profit_factor: Optional[float] = None
    avg_trade_return: float

    # Exposure
    avg_long_exposure: float
    avg_short_exposure: float
    avg_gross_exposure: float
    avg_net_exposure: float

    # Costs
    total_commission: float
    total_slippage: float
    total_funding_fees: float
    total_leverage_fees: float


class PositionSnapshot(BaseModel):
    """Position at a point in time."""
    timestamp: datetime
    asset: str
    direction: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float


class BacktestResult(BaseModel):
    """Complete backtest result."""
    task_id: str
    status: TaskStatus

    # Configuration
    config: Dict[str, Any]

    # Performance
    metrics: PerformanceMetrics

    # Time series data
    nav_series: List[Dict[str, Any]]  # [{"timestamp": ..., "nav": ...}, ...]
    drawdown_series: List[Dict[str, Any]]

    # Trades
    trades: List[TradeRecord]

    # Final positions
    final_positions: List[PositionSnapshot]

    # Metadata
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
