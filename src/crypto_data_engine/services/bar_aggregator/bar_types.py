"""
Bar type definitions and builders.

Supports multiple bar types:
- TimeBar: Fixed time intervals (1min, 5min, 1h, 1d, etc.)
- TickBar: Fixed number of ticks
- VolumeBar: Fixed volume threshold
- DollarBar: Fixed dollar volume threshold

Each bar type produces a standardized set of features.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


class BarType(Enum):
    """Supported bar types."""
    TIME_BAR = "time_bar"
    TICK_BAR = "tick_bar"
    VOLUME_BAR = "volume_bar"
    DOLLAR_BAR = "dollar_bar"


@dataclass
class BarConfig:
    """Configuration for bar construction."""
    bar_type: BarType
    threshold: Union[int, float, str]  # count, volume, dollars, or time string
    include_advanced_features: bool = True
    min_ticks: int = 1  # Minimum ticks for a valid bar
    
    def __post_init__(self):
        if isinstance(self.bar_type, str):
            self.bar_type = BarType(self.bar_type)


class BaseBarBuilder(ABC):
    """
    Abstract base class for bar builders.
    
    All bar builders produce a standardized set of features.
    """
    
    # Standard columns expected in tick data
    TICK_COLUMNS = [
        "timestamp", "price", "quantity", "isBuyerMaker"
    ]
    
    # Basic features (all bar types)
    BASIC_FEATURES = [
        "start_time", "end_time", "open", "high", "low", "close",
        "volume", "buy_volume", "sell_volume", "vwap",
        "tick_count", "dollar_volume",
    ]
    
    # Advanced features (optional)
    ADVANCED_FEATURES = [
        "price_std", "volume_std", "up_move_ratio", "down_move_ratio",
        "reversals", "buy_sell_imbalance", "spread_proxy",
        "skewness", "kurtosis", "max_trade_volume", "max_trade_ratio",
        "tick_interval_mean", "tick_interval_std",
    ]
    
    def __init__(self, config: BarConfig):
        self.config = config
    
    @abstractmethod
    def get_split_indices(self, data: pd.DataFrame) -> List[int]:
        """
        Get indices where bars should be split.
        
        Args:
            data: Tick data DataFrame
        
        Returns:
            List of indices marking bar boundaries
        """
        pass
    
    def build_bars(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build bars from tick data.
        
        Args:
            data: Tick data with columns [timestamp, price, quantity, isBuyerMaker]
        
        Returns:
            DataFrame of bars with standardized features
        """
        # Validate input
        data = self._validate_and_prepare(data)
        
        # Get split points
        split_indices = self.get_split_indices(data)
        
        # Build bars
        bars = []
        start_idx = 0
        
        for end_idx in split_indices:
            if end_idx <= start_idx:
                continue
            
            segment = data.iloc[start_idx:end_idx + 1]
            
            if len(segment) >= self.config.min_ticks:
                bar = self._build_single_bar(segment)
                bars.append(bar)
            
            start_idx = end_idx + 1
        
        # Handle remaining data
        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            if len(segment) >= self.config.min_ticks:
                bar = self._build_single_bar(segment)
                bars.append(bar)
        
        if not bars:
            return pd.DataFrame(columns=self._get_all_columns())
        
        df = pd.DataFrame(bars)
        return df
    
    def _validate_and_prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare tick data.

        Avoids a full ``copy()`` unless we actually need to mutate the
        DataFrame (i.e. add a missing column).  Sorting is done inplace
        on the (possibly shared) frame; callers that need the original
        order should copy before calling.
        """
        # Ensure required columns exist
        required = ["timestamp", "price", "quantity"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Only copy if we need to add isBuyerMaker (avoids 2x memory)
        needs_buyer_col = "isBuyerMaker" not in data.columns
        if needs_buyer_col:
            data = data.copy()
            data["isBuyerMaker"] = False

        # Inplace sort avoids creating another full-size DataFrame
        if not data["timestamp"].is_monotonic_increasing:
            data = data.sort_values("timestamp", ignore_index=True)

        return data
    
    def _build_single_bar(self, segment: pd.DataFrame) -> Dict[str, Any]:
        """Build a single bar from a segment of tick data."""
        bar = self._build_basic_features(segment)
        
        if self.config.include_advanced_features:
            advanced = self._build_advanced_features(segment)
            bar.update(advanced)
        
        return bar
    
    def _build_basic_features(self, segment: pd.DataFrame) -> Dict[str, Any]:
        """Build basic OHLCV features."""
        price = segment["price"]
        qty = segment["quantity"]
        is_buyer = segment["isBuyerMaker"]
        
        buy_volume = qty[~is_buyer].sum()
        sell_volume = qty[is_buyer].sum()
        total_volume = qty.sum()
        
        return {
            "start_time": self._convert_timestamp(segment["timestamp"].iloc[0]),
            "end_time": self._convert_timestamp(segment["timestamp"].iloc[-1]),
            "open": price.iloc[0],
            "high": price.max(),
            "low": price.min(),
            "close": price.iloc[-1],
            "volume": total_volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "vwap": (price * qty).sum() / total_volume if total_volume > 0 else price.mean(),
            "tick_count": len(segment),
            "dollar_volume": (price * qty).sum(),
        }
    
    def _build_advanced_features(self, segment: pd.DataFrame) -> Dict[str, Any]:
        """Build advanced microstructure features."""
        price = segment["price"]
        qty = segment["quantity"]
        is_buyer = segment["isBuyerMaker"]
        ts = segment["timestamp"]
        
        # Price movement
        price_diff = price.diff()
        up_moves = (price_diff > 0).sum()
        down_moves = (price_diff < 0).sum()
        total_moves = up_moves + down_moves
        
        # Direction changes (reversals)
        direction = np.sign(price_diff)
        reversals = np.count_nonzero(direction.diff().fillna(0))
        
        # Returns for higher moments
        returns = price.pct_change().dropna()
        
        # Time intervals
        ts_diff = ts.diff().dropna()
        
        # Maximum trade
        max_idx = qty.idxmax()
        max_vol = qty.loc[max_idx]
        
        # Buy-sell imbalance
        buy_vol = qty[~is_buyer].sum()
        sell_vol = qty[is_buyer].sum()
        total_vol = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
        
        # Path Efficiency: |close - open| / sum(|tick-to-tick price changes|)
        abs_price_diff = price_diff.abs()
        total_path_length = abs_price_diff.sum()
        abs_net_move = abs(price.iloc[-1] - price.iloc[0])
        path_efficiency = abs_net_move / total_path_length if total_path_length > 0 else 0.0

        # Intrabar Impact Density: |close - open| / dollar_volume
        dollar_volume = (price * qty).sum()
        impact_density = abs_net_move / dollar_volume if dollar_volume > 0 else 0.0

        return {
            "price_std": price.std(),
            "volume_std": qty.std(),
            "up_move_ratio": up_moves / total_moves if total_moves > 0 else 0.5,
            "down_move_ratio": down_moves / total_moves if total_moves > 0 else 0.5,
            "reversals": reversals,
            "buy_sell_imbalance": imbalance,
            "spread_proxy": (price.max() - price.min()) / price.mean() if price.mean() > 0 else 0,
            "skewness": skew(returns) if len(returns) > 2 else np.nan,
            "kurtosis": kurtosis(returns) if len(returns) > 3 else np.nan,
            "max_trade_volume": max_vol,
            "max_trade_ratio": max_vol / qty.sum() if qty.sum() > 0 else 0,
            "tick_interval_mean": ts_diff.mean() if len(ts_diff) > 0 else 0,
            "tick_interval_std": ts_diff.std() if len(ts_diff) > 1 else 0,
            "path_efficiency": path_efficiency,
            "impact_density": impact_density,
        }
    
    def _convert_timestamp(self, ts: Union[int, float, datetime]) -> datetime:
        """Convert timestamp to datetime."""
        if isinstance(ts, datetime):
            return ts
        
        # Handle different timestamp formats
        ts_int = int(ts)
        length = len(str(ts_int))
        
        if length >= 16:  # microseconds
            ts_sec = ts / 1_000_000
        elif length == 13:  # milliseconds
            ts_sec = ts / 1_000
        else:  # seconds
            ts_sec = ts
        
        return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    
    def _get_all_columns(self) -> List[str]:
        """Get all output columns."""
        cols = self.BASIC_FEATURES.copy()
        if self.config.include_advanced_features:
            cols.extend(self.ADVANCED_FEATURES)
        return cols


class TimeBarBuilder(BaseBarBuilder):
    """
    Build time bars (fixed time intervals).
    
    Example:
        builder = TimeBarBuilder(BarConfig(BarType.TIME_BAR, "1min"))
        bars = builder.build_bars(tick_data)
    """
    
    # Common interval mappings
    INTERVAL_MAP = {
        "1s": pd.Timedelta(seconds=1),
        "5s": pd.Timedelta(seconds=5),
        "10s": pd.Timedelta(seconds=10),
        "30s": pd.Timedelta(seconds=30),
        "1min": pd.Timedelta(minutes=1),
        "5min": pd.Timedelta(minutes=5),
        "15min": pd.Timedelta(minutes=15),
        "30min": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
        "1w": pd.Timedelta(weeks=1),
    }
    
    def __init__(self, config: BarConfig):
        super().__init__(config)
        self.interval = self._parse_interval(config.threshold)
    
    def _parse_interval(self, interval: Union[str, pd.Timedelta]) -> pd.Timedelta:
        """Parse interval string to Timedelta."""
        if isinstance(interval, pd.Timedelta):
            return interval
        
        if interval in self.INTERVAL_MAP:
            return self.INTERVAL_MAP[interval]
        
        # Try pandas Timedelta parsing
        try:
            return pd.Timedelta(interval)
        except ValueError:
            raise ValueError(f"Invalid interval: {interval}")
    
    def get_split_indices(self, data: pd.DataFrame) -> List[int]:
        """Get indices where time bars should be split.

        Uses vectorised numpy operations instead of a Python for-loop,
        which is ~50-100x faster for large tick datasets.
        """
        timestamps = pd.to_datetime(data["timestamp"], unit="ms")

        # Floor timestamps to interval
        floored = timestamps.dt.floor(self.interval).values  # numpy array

        # Vectorised: find positions where the floored value changes
        change_mask = floored[1:] != floored[:-1]
        # Indices of the *last* tick in each bar = positions just before a change
        boundary_indices = np.nonzero(change_mask)[0]  # these are end-of-bar indices

        # Build final list: boundary indices + the very last index
        indices = boundary_indices.tolist()
        if len(data) > 0:
            indices.append(len(data) - 1)

        return indices


class TickBarBuilder(BaseBarBuilder):
    """
    Build tick bars (fixed number of ticks).
    
    Example:
        builder = TickBarBuilder(BarConfig(BarType.TICK_BAR, 1000))
        bars = builder.build_bars(tick_data)
    """
    
    def get_split_indices(self, data: pd.DataFrame) -> List[int]:
        """Get indices where tick bars should be split."""
        threshold = int(self.config.threshold)
        indices = list(range(threshold - 1, len(data), threshold))
        
        # Ensure last bar is included
        if indices and indices[-1] != len(data) - 1:
            indices.append(len(data) - 1)
        
        return indices


class VolumeBarBuilder(BaseBarBuilder):
    """
    Build volume bars (fixed volume threshold).
    
    Example:
        builder = VolumeBarBuilder(BarConfig(BarType.VOLUME_BAR, 100000))
        bars = builder.build_bars(tick_data)
    """
    
    def get_split_indices(self, data: pd.DataFrame) -> List[int]:
        """Get indices where volume bars should be split."""
        threshold = float(self.config.threshold)
        volume = data["quantity"].values
        
        indices = []
        cum_volume = 0.0
        
        for i, v in enumerate(volume):
            cum_volume += v
            if cum_volume >= threshold:
                indices.append(i)
                cum_volume = 0.0
        
        # Add final bar
        if len(data) > 0 and (not indices or indices[-1] != len(data) - 1):
            indices.append(len(data) - 1)
        
        return indices


class DollarBarBuilder(BaseBarBuilder):
    """
    Build dollar bars (fixed dollar volume threshold).
    
    Example:
        builder = DollarBarBuilder(BarConfig(BarType.DOLLAR_BAR, 1000000))
        bars = builder.build_bars(tick_data)
    """
    
    def get_split_indices(self, data: pd.DataFrame) -> List[int]:
        """Get indices where dollar bars should be split."""
        threshold = float(self.config.threshold)
        dollar_volume = (data["price"] * data["quantity"]).values
        
        indices = []
        cum_dollar = 0.0
        
        for i, dv in enumerate(dollar_volume):
            cum_dollar += dv
            if cum_dollar >= threshold:
                indices.append(i)
                cum_dollar = 0.0
        
        # Add final bar
        if len(data) > 0 and (not indices or indices[-1] != len(data) - 1):
            indices.append(len(data) - 1)
        
        return indices


def get_bar_builder(config: BarConfig) -> BaseBarBuilder:
    """
    Factory function to get the appropriate bar builder.
    
    Args:
        config: Bar configuration
    
    Returns:
        Appropriate bar builder instance
    """
    builders = {
        BarType.TIME_BAR: TimeBarBuilder,
        BarType.TICK_BAR: TickBarBuilder,
        BarType.VOLUME_BAR: VolumeBarBuilder,
        BarType.DOLLAR_BAR: DollarBarBuilder,
    }
    
    builder_class = builders.get(config.bar_type)
    if builder_class is None:
        raise ValueError(f"Unknown bar type: {config.bar_type}")
    
    return builder_class(config)


def build_bars(
    tick_data: pd.DataFrame,
    bar_type: Union[BarType, str],
    threshold: Union[int, float, str],
    include_advanced: bool = True
) -> pd.DataFrame:
    """
    Convenience function to build bars from tick data.
    
    Args:
        tick_data: Tick data DataFrame
        bar_type: Type of bars to build
        threshold: Bar threshold (count, volume, dollars, or time string)
        include_advanced: Whether to include advanced features
    
    Returns:
        DataFrame of bars
    """
    if isinstance(bar_type, str):
        bar_type = BarType(bar_type)
    
    config = BarConfig(
        bar_type=bar_type,
        threshold=threshold,
        include_advanced_features=include_advanced,
    )
    
    builder = get_bar_builder(config)
    return builder.build_bars(tick_data)
