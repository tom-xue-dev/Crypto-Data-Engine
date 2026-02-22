"""
Base classes for the backtest system.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


class TradeDirection(Enum):
    """Trade direction enum."""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"


class SignalType(Enum):
    """Signal type enum."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeRecord:
    """Record of a single trade."""
    asset: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None
    
    @property
    def holding_period(self) -> Optional[pd.Timedelta]:
        if self.exit_time is None:
            return None
        return pd.Timedelta(self.exit_time - self.entry_time)


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time."""
    timestamp: datetime
    cash: float
    positions: Dict[str, float]  # asset -> quantity
    prices: Dict[str, float]  # asset -> price
    nav: float  # Net Asset Value
    
    @property
    def market_value(self) -> float:
        """Calculate total market value of positions."""
        return sum(
            qty * self.prices.get(asset, 0)
            for asset, qty in self.positions.items()
        )


@dataclass
class BacktestResult:
    """Complete backtest result."""
    # Basic info
    start_time: datetime
    end_time: datetime
    initial_capital: float
    final_capital: float
    
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0  # hours
    
    # Time series
    nav_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[TradeRecord] = field(default_factory=list)
    portfolio_history: List[PortfolioSnapshot] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
        }


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Subclasses must implement either:
    - generate_signals() for time-series strategies
    - generate_weights() for cross-sectional strategies
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self._params: Dict[str, Any] = {}
    
    @property
    def params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self._params.copy()
    
    def set_params(self, **kwargs) -> None:
        """Set strategy parameters."""
        self._params.update(kwargs)
    
    def generate_signal(
        self, 
        bar_data: pd.Series, 
        position: Optional[float] = None
    ) -> SignalType:
        """
        Generate trading signal for a single asset (time-series mode).
        
        Args:
            bar_data: Current bar data for the asset
            position: Current position quantity (positive=long, negative=short)
        
        Returns:
            SignalType indicating the trading action
        """
        raise NotImplementedError("Subclass must implement generate_signal for time-series mode")
    
    def generate_weights(
        self, 
        cross_section: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Generate target portfolio weights (cross-sectional mode).
        
        Args:
            cross_section: DataFrame with assets as rows, features as columns
            current_weights: Current portfolio weights
        
        Returns:
            Dict mapping asset to target weight (weights should sum to <= 1.0)
        """
        raise NotImplementedError("Subclass must implement generate_weights for cross-sectional mode")
    
    def on_trade(self, trade: TradeRecord) -> None:
        """
        Callback when a trade is executed.
        
        Override this method to implement custom logic after trades.
        
        Args:
            trade: The executed trade record
        """
        pass
    
    def on_period_end(self, timestamp: datetime, portfolio: PortfolioSnapshot) -> None:
        """
        Callback at the end of each period.
        
        Override this method to implement custom period-end logic.
        
        Args:
            timestamp: Current timestamp
            portfolio: Current portfolio state
        """
        pass


class FeatureCalculator(ABC):
    """Abstract base class for feature calculators."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features from bar data.
        
        Args:
            data: Input bar data
        
        Returns:
            DataFrame with calculated features
        """
        pass


class RollingFeatureCalculator(FeatureCalculator):
    """Calculator for rolling window features."""
    
    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 20, 60, 120]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling features."""
        df = data.copy()
        
        # Returns
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        
        for window in self.windows:
            # Rolling returns
            df[f"return_{window}"] = df["close"].pct_change(window)
            
            # Rolling mean and std
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"std_{window}"] = df["close"].rolling(window).std()
            
            # Rolling volume
            df[f"vol_sma_{window}"] = df["volume"].rolling(window).mean()
            
            # Momentum
            df[f"momentum_{window}"] = df["close"] / df["close"].shift(window) - 1
            
            # Volatility
            df[f"volatility_{window}"] = df["return"].rolling(window).std() * np.sqrt(252)
        
        return df


def calculate_performance_metrics(
    nav_series: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Calculate performance metrics from NAV series.

    Uses calendar-time aware annualization:
    - Annual return based on elapsed time between first/last timestamps
    - Sharpe/Sortino computed on daily returns (resampled) when possible
    """
    if nav_series is None or len(nav_series) < 2:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    # Ensure datetime index and sorted
    s = nav_series.copy()
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    s = s.sort_index()

    # Total and annualized return by elapsed time
    total_return = float(s.iloc[-1] / s.iloc[0] - 1.0)
    elapsed_years = max((s.index[-1] - s.index[0]).total_seconds() / (365.25 * 24 * 3600), 0.0) 
    annual_return = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0 if elapsed_years > 0 else 0.0

    # Daily returns for risk ratios when timestamps are datetime-like
    try:
        daily_nav = s.resample("1D").last().dropna()
        daily_rets = daily_nav.pct_change().dropna()
    except Exception:
        daily_rets = s.pct_change().dropna()

    if len(daily_rets) < 2:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        max_drawdown = 0.0
    else:
        excess = daily_rets - risk_free_rate / periods_per_year
        vol = excess.std(ddof=0)
        sharpe_ratio = float(np.sqrt(periods_per_year) * excess.mean() / vol) if vol > 0 else 0.0

        downside = excess[excess < 0]
        downside_vol = downside.std(ddof=0) if len(downside) > 0 else 0.0
        sortino_ratio = float(np.sqrt(periods_per_year) * excess.mean() / downside_vol) if downside_vol > 0 else 0.0

        # Max drawdown on daily series
        cumulative = (1.0 + daily_rets).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(abs(drawdown.min())) if len(drawdown) else 0.0

    calmar_ratio = float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }
