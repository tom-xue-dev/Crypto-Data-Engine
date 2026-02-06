"""
Core module for the crypto data engine.
Contains base classes, interfaces, and common utilities.
"""

from .base import (
    BaseStrategy,
    BacktestResult,
    TradeRecord,
    TradeDirection,
    SignalType,
    PortfolioSnapshot,
    calculate_performance_metrics,
)
from .interfaces import (
    IPortfolio,
    IOrderExecutor,
    IRiskManager,
    ICostModel,
    IDataLoader,
    IBacktestEngine,
    Order,
    OrderSide,
    OrderStatus,
    PositionInfo,
    PositionType,
    CloseSignal,
    PositionKey,
    PriceMap,
    WeightMap,
)

__all__ = [
    # Base classes
    "BaseStrategy",
    "BacktestResult",
    "TradeRecord",
    "TradeDirection",
    "SignalType",
    "PortfolioSnapshot",
    "calculate_performance_metrics",
    # Interfaces
    "IPortfolio",
    "IOrderExecutor",
    "IRiskManager",
    "ICostModel",
    "IDataLoader",
    "IBacktestEngine",
    # Data classes
    "Order",
    "OrderSide",
    "OrderStatus",
    "PositionInfo",
    "PositionType",
    "CloseSignal",
    # Type aliases
    "PositionKey",
    "PriceMap",
    "WeightMap",
]
