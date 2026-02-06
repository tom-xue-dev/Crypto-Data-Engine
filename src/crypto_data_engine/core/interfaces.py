"""
Core interfaces for the backtest and simulation system.

These interfaces define contracts that are shared between backtest and live trading,
enabling code reuse and consistent behavior across environments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd

from crypto_data_engine.core.base import TradeDirection, TradeRecord


class OrderSide(Enum):
    """Order side enum."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionType(Enum):
    """Position type for different market types."""
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURE = "future"


@dataclass
class Order:
    """Order data class."""
    order_id: str
    asset: str
    side: OrderSide
    quantity: float
    price: float
    order_type: str = "market"  # market, limit
    status: OrderStatus = OrderStatus.PENDING
    timestamp: int = 0  # ms
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    filled_timestamp: int = 0
    direction: TradeDirection = TradeDirection.LONG
    leverage: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "asset": self.asset,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "direction": self.direction.value,
            "leverage": self.leverage,
        }


@dataclass
class PositionInfo:
    """Position information shared between backtest and live."""
    asset: str
    direction: TradeDirection
    quantity: float
    entry_price: float
    entry_time: datetime
    leverage: float = 1.0
    position_type: PositionType = PositionType.SPOT
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Total market value of position."""
        return self.quantity * self.current_price

    @property
    def own_equity(self) -> float:
        """Self-owned equity (excluding leveraged portion)."""
        return self.entry_price * self.quantity / self.leverage

    @property
    def margin(self) -> float:
        """Required margin for position."""
        return self.market_value / self.leverage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "leverage": self.leverage,
            "position_type": self.position_type.value,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "market_value": self.market_value,
        }


@dataclass
class CloseSignal:
    """Signal to close a position."""
    asset: str
    direction: TradeDirection
    reason: str
    quantity: Optional[float] = None  # None means close all
    timestamp: Optional[datetime] = None


class IPortfolio(ABC):
    """Interface for portfolio management."""

    @property
    @abstractmethod
    def cash(self) -> float:
        """Available cash balance."""
        pass

    @property
    @abstractmethod
    def nav(self) -> float:
        """Net Asset Value."""
        pass

    @property
    @abstractmethod
    def leverage(self) -> float:
        """Current portfolio leverage."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[Tuple[str, TradeDirection], PositionInfo]:
        """Get all positions."""
        pass

    @abstractmethod
    def get_position(self, asset: str, direction: TradeDirection) -> Optional[PositionInfo]:
        """Get position for specific asset and direction."""
        pass

    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        pass

    @abstractmethod
    def update_prices(self, price_map: Dict[str, float]) -> None:
        """Update all position prices."""
        pass

    @abstractmethod
    def calculate_nav(self, price_map: Dict[str, float]) -> float:
        """Calculate NAV with given prices."""
        pass


class IOrderExecutor(ABC):
    """Interface for order execution."""

    @abstractmethod
    def create_order(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        price: float,
        direction: TradeDirection = TradeDirection.LONG,
        leverage: float = 1.0,
        **kwargs
    ) -> Order:
        """Create a new order."""
        pass

    @abstractmethod
    def execute_order(self, order: Order) -> bool:
        """Execute an order. Returns True if successful."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if successful."""
        pass


class IRiskManager(ABC):
    """Interface for risk management."""

    @abstractmethod
    def check_position_risk(
        self,
        portfolio: IPortfolio,
        asset: str,
        direction: TradeDirection,
        price: float,
        timestamp: datetime
    ) -> List[CloseSignal]:
        """Check risk for a specific position."""
        pass

    @abstractmethod
    def check_portfolio_risk(
        self,
        portfolio: IPortfolio,
        price_map: Dict[str, float],
        timestamp: datetime
    ) -> List[CloseSignal]:
        """Check overall portfolio risk."""
        pass

    @abstractmethod
    def check_order_risk(
        self,
        portfolio: IPortfolio,
        order: Order,
        current_nav: float
    ) -> Tuple[bool, str]:
        """
        Check if order passes risk checks.
        Returns (is_allowed, reason).
        """
        pass


class ICostModel(ABC):
    """Interface for cost calculation."""

    @abstractmethod
    def calculate_commission(self, value: float, is_maker: bool = False) -> float:
        """Calculate commission for a trade."""
        pass

    @abstractmethod
    def apply_slippage(self, price: float, direction: TradeDirection) -> float:
        """Apply slippage to execution price."""
        pass

    @abstractmethod
    def calculate_total_cost(
        self,
        value: float,
        direction: TradeDirection,
        is_maker: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate total execution cost.
        Returns (execution_price_adjustment, total_fees).
        """
        pass

    @abstractmethod
    def settle_periodic_fees(
        self,
        portfolio: IPortfolio,
        timestamp: datetime,
        price_map: Dict[str, float]
    ) -> float:
        """Settle periodic fees (funding, leverage). Returns total fees charged."""
        pass


class IDataLoader(ABC):
    """Interface for data loading."""

    @abstractmethod
    def load_bars(
        self,
        assets: List[str],
        start_date: datetime,
        end_date: datetime,
        bar_type: str = "time"
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load bar data for assets.

        Returns:
            For aligned data: DataFrame with MultiIndex (time, asset)
            For non-aligned data: Dict[asset, DataFrame]
        """
        pass

    @abstractmethod
    def load_features(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Calculate and return features for the data."""
        pass


class IBacktestEngine(ABC):
    """Interface for backtest engine."""

    @abstractmethod
    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> "BacktestResult":
        """Run the backtest and return results."""
        pass

    @abstractmethod
    def step(
        self,
        timestamp: datetime,
        bars: Dict[str, pd.Series]
    ) -> None:
        """Process a single time step."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        pass


# Type aliases for convenience
PositionKey = Tuple[str, TradeDirection]
PriceMap = Dict[str, float]
WeightMap = Dict[str, float]
