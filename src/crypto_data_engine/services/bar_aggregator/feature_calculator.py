"""
Feature calculator for bar data.

Provides rolling window features and cross-sectional features
for use in backtesting and live trading.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature calculation."""
    # Rolling windows
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120, 240])
    
    # Feature groups to calculate
    calc_returns: bool = True
    calc_volatility: bool = True
    calc_momentum: bool = True
    calc_volume: bool = True
    calc_microstructure: bool = True
    
    # Cross-sectional features
    calc_cross_sectional: bool = False
    
    # Output
    drop_na: bool = False
    prefix: str = ""


class BaseFeatureCalculator(ABC):
    """Abstract base class for feature calculators."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from bar data."""
        pass


class RollingFeatureCalculator(BaseFeatureCalculator):
    """
    Calculator for rolling window features.
    
    Features include:
    - Returns (simple and log)
    - Moving averages (SMA, EMA)
    - Volatility
    - Momentum indicators
    - Volume features
    - Microstructure features
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured features.
        
        Args:
            data: Bar data with OHLCV columns
        
        Returns:
            DataFrame with original data and calculated features
        """
        df = data.copy()
        prefix = self.config.prefix
        
        if self.config.calc_returns:
            df = self._calc_returns(df, prefix)
        
        if self.config.calc_volatility:
            df = self._calc_volatility(df, prefix)
        
        if self.config.calc_momentum:
            df = self._calc_momentum(df, prefix)
        
        if self.config.calc_volume:
            df = self._calc_volume_features(df, prefix)
        
        if self.config.calc_microstructure:
            df = self._calc_microstructure(df, prefix)
        
        if self.config.drop_na:
            df = df.dropna()
        
        return df
    
    def _calc_returns(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Calculate return features."""
        if "close" not in df.columns:
            return df
        
        # Simple return
        df[f"{prefix}return_1"] = df["close"].pct_change()
        df[f"{prefix}log_return_1"] = np.log(df["close"] / df["close"].shift(1))
        
        for window in self.config.windows:
            # Multi-period returns
            df[f"{prefix}return_{window}"] = df["close"].pct_change(window)
            df[f"{prefix}log_return_{window}"] = np.log(df["close"] / df["close"].shift(window))
            
            # Cumulative returns
            df[f"{prefix}cum_return_{window}"] = df[f"{prefix}return_1"].rolling(window).sum()
        
        return df
    
    def _calc_volatility(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Calculate volatility features."""
        if "close" not in df.columns:
            return df
        
        # Calculate return if not present
        if f"{prefix}return_1" not in df.columns:
            df[f"{prefix}return_1"] = df["close"].pct_change()
        
        for window in self.config.windows:
            # Rolling standard deviation
            df[f"{prefix}volatility_{window}"] = df[f"{prefix}return_1"].rolling(window).std()
            
            # Annualized volatility (assuming daily bars)
            df[f"{prefix}ann_volatility_{window}"] = df[f"{prefix}volatility_{window}"] * np.sqrt(252)
            
            # Parkinson volatility (using high/low)
            if "high" in df.columns and "low" in df.columns:
                log_hl = np.log(df["high"] / df["low"])
                df[f"{prefix}parkinson_vol_{window}"] = (
                    log_hl.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2))))
                )
        
        return df
    
    def _calc_momentum(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Calculate momentum features."""
        if "close" not in df.columns:
            return df
        
        for window in self.config.windows:
            # Price momentum
            df[f"{prefix}momentum_{window}"] = df["close"] / df["close"].shift(window) - 1
            
            # Rate of change
            df[f"{prefix}roc_{window}"] = (df["close"] - df["close"].shift(window)) / df["close"].shift(window) * 100
            
            # Moving averages
            df[f"{prefix}sma_{window}"] = df["close"].rolling(window).mean()
            df[f"{prefix}ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
            
            # Price relative to SMA
            df[f"{prefix}price_sma_ratio_{window}"] = df["close"] / df[f"{prefix}sma_{window}"]
            
            # Bollinger Band position
            bb_std = df["close"].rolling(window).std()
            bb_upper = df[f"{prefix}sma_{window}"] + 2 * bb_std
            bb_lower = df[f"{prefix}sma_{window}"] - 2 * bb_std
            df[f"{prefix}bb_position_{window}"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        for window in self.config.windows:
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f"{prefix}rsi_{window}"] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calc_volume_features(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Calculate volume features."""
        if "volume" not in df.columns:
            return df
        
        for window in self.config.windows:
            # Volume moving average
            df[f"{prefix}volume_sma_{window}"] = df["volume"].rolling(window).mean()
            
            # Volume ratio (current vs average)
            df[f"{prefix}volume_ratio_{window}"] = df["volume"] / df[f"{prefix}volume_sma_{window}"]
            
            # Volume standard deviation
            df[f"{prefix}volume_std_{window}"] = df["volume"].rolling(window).std()
            
            # VWAP deviation
            if "vwap" in df.columns:
                df[f"{prefix}vwap_deviation_{window}"] = (df["close"] - df["vwap"]) / df["vwap"]
        
        # Buy/sell volume features
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            df[f"{prefix}buy_ratio"] = df["buy_volume"] / df["volume"]
            df[f"{prefix}order_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / df["volume"]
            
            for window in self.config.windows:
                df[f"{prefix}buy_ratio_sma_{window}"] = df[f"{prefix}buy_ratio"].rolling(window).mean()
                df[f"{prefix}order_imbalance_sma_{window}"] = df[f"{prefix}order_imbalance"].rolling(window).mean()
        
        return df
    
    def _calc_microstructure(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Calculate microstructure features."""
        for window in self.config.windows:
            # Spread proxy
            if "high" in df.columns and "low" in df.columns:
                df[f"{prefix}spread_proxy_{window}"] = (
                    (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2)
                ).rolling(window).mean()
            
            # Tick features
            if "tick_count" in df.columns:
                df[f"{prefix}tick_intensity_{window}"] = df["tick_count"].rolling(window).mean()
            
            # Reversal features
            if "reversals" in df.columns:
                df[f"{prefix}reversal_rate_{window}"] = df["reversals"].rolling(window).mean()
            
            # Amihud illiquidity
            if f"{prefix}return_1" in df.columns and "dollar_volume" in df.columns:
                amihud = df[f"{prefix}return_1"].abs() / df["dollar_volume"].replace(0, np.nan)
                df[f"{prefix}amihud_{window}"] = amihud.rolling(window).mean()
            
            # Path Efficiency: rolling mean captures trending vs mean-reverting regime
            if "path_efficiency" in df.columns:
                df[f"{prefix}path_efficiency_{window}"] = (
                    df["path_efficiency"].rolling(window).mean()
                )

            # Signed Path Efficiency: PE × sign(bar return)
            if f"{prefix}signed_path_efficiency" not in df.columns:
                if "path_efficiency" in df.columns and "close" in df.columns:
                    bar_sign = np.sign(df["close"] - df["open"]) if "open" in df.columns else np.sign(df["close"].diff())
                    df[f"{prefix}signed_path_efficiency"] = df["path_efficiency"] * bar_sign
            if f"{prefix}signed_path_efficiency" in df.columns:
                df[f"{prefix}signed_pe_{window}"] = (
                    df[f"{prefix}signed_path_efficiency"].rolling(window).mean()
                )

            # Intrabar Impact Density: rolling mean captures liquidity regime
            if "impact_density" in df.columns:
                df[f"{prefix}impact_density_{window}"] = (
                    df["impact_density"].rolling(window).mean()
                )
        
        return df


class CrossSectionalFeatureCalculator(BaseFeatureCalculator):
    """
    Calculator for cross-sectional features.
    
    Computes features that compare assets at the same point in time,
    such as percentile ranks, z-scores, and relative metrics.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional features.
        
        Args:
            data: Bar data with MultiIndex (time, asset)
        
        Returns:
            DataFrame with cross-sectional features
        """
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Cross-sectional calculation requires MultiIndex (time, asset)")
        
        df = data.copy()
        
        # Columns to rank
        rank_cols = ["return_1", "momentum_20", "volume", "volatility_20"]
        rank_cols = [c for c in rank_cols if c in df.columns]
        
        for col in rank_cols:
            # Percentile rank within each time period
            df[f"{col}_rank"] = df.groupby(level="time")[col].transform(
                lambda x: x.rank(pct=True)
            )
            
            # Z-score within each time period
            df[f"{col}_zscore"] = df.groupby(level="time")[col].transform(
                lambda x: (x - x.mean()) / x.std()
            )
        
        return df


class FeaturePipeline:
    """
    Pipeline for chaining multiple feature calculators.
    
    Usage:
        pipeline = FeaturePipeline([
            RollingFeatureCalculator(config),
            CrossSectionalFeatureCalculator(config),
        ])
        features = pipeline.calculate(data)
    """
    
    def __init__(self, calculators: List[BaseFeatureCalculator]):
        self.calculators = calculators
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run all calculators in sequence."""
        df = data.copy()
        for calculator in self.calculators:
            df = calculator.calculate(df)
        return df
    
    def add_calculator(self, calculator: BaseFeatureCalculator) -> None:
        """Add a calculator to the pipeline."""
        self.calculators.append(calculator)


def calculate_weekly_returns(
    data: pd.DataFrame,
    price_col: str = "close",
    time_col: str = "start_time"
) -> pd.DataFrame:
    """
    Calculate weekly returns for cross-sectional strategies.
    
    Args:
        data: Bar data with MultiIndex (time, asset) or time column
        price_col: Column containing price
        time_col: Column containing timestamp
    
    Returns:
        DataFrame with weekly returns per asset
    """
    df = data.copy()
    
    # Handle MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    
    # Ensure datetime
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df["week"] = df[time_col].dt.to_period("W")
    else:
        raise ValueError(f"Time column '{time_col}' not found")
    
    # Calculate weekly returns per asset
    weekly = df.groupby(["asset", "week"]).agg({
        price_col: ["first", "last"]
    })
    weekly.columns = ["open_price", "close_price"]
    weekly["weekly_return"] = (weekly["close_price"] / weekly["open_price"]) - 1
    
    return weekly.reset_index()


def calculate_monthly_turnover(
    data: pd.DataFrame,
    time_col: str = "start_time"
) -> pd.DataFrame:
    """
    Calculate monthly turnover (dollar volume) per asset.
    
    Args:
        data: Bar data with MultiIndex (time, asset) or asset column
        time_col: Column containing timestamp
    
    Returns:
        DataFrame with monthly turnover per asset
    """
    df = data.copy()
    
    # Handle MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    
    # Ensure datetime
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df["month"] = df[time_col].dt.to_period("M")
    else:
        raise ValueError(f"Time column '{time_col}' not found")
    
    # Calculate dollar volume if not present
    if "dollar_volume" not in df.columns:
        if "volume" in df.columns and "vwap" in df.columns:
            df["dollar_volume"] = df["volume"] * df["vwap"]
        elif "volume" in df.columns and "close" in df.columns:
            df["dollar_volume"] = df["volume"] * df["close"]
        else:
            raise ValueError("Cannot calculate dollar volume: missing columns")
    
    # Aggregate by asset and month
    monthly = df.groupby(["asset", "month"]).agg({
        "dollar_volume": "sum",
        "volume": "sum",
        "close": "last"
    }).reset_index()
    
    monthly.columns = ["asset", "month", "monthly_turnover", "monthly_volume", "month_end_price"]
    
    return monthly
