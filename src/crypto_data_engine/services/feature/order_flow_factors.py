"""
Order Flow Factor Calculation Module.

Implements factors from the quantitative trading strategy design:
- OFI (Order Flow Imbalance): Multi-scale windows [20, 100, 500]
- SmartFlow (Large Order Flow): Windows [50, 200]
- Trade Intensity (λ): Window 20 with z-score normalization
- VPIN (Volume-Synchronized Probability of Informed Trading): Window 50
- Kyle Lambda (Price Impact): Window 100
- MR Signal (Mean Reversion): Fast=20, Slow=200
- Sweep Detection: 100ms window, 5+ trades, directional price movement

All factors are calculated on Dollar Bar data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class OrderFlowFactorConfig:
    """Configuration for Order Flow factor calculation."""

    # OFI windows
    ofi_fast_window: int = 20
    ofi_mid_window: int = 100
    ofi_slow_window: int = 500

    # SmartFlow
    smart_flow_windows: List[int] = field(default_factory=lambda: [50, 200])
    big_trade_lookback: int = 2000  # Bars for calculating big trade threshold
    big_trade_multiplier: float = 5.0  # Threshold = mean * multiplier

    # Trade Intensity
    lambda_window: int = 20
    lambda_zscore_window: int = 500

    # VPIN
    vpin_window: int = 50

    # Kyle Lambda
    kyle_lambda_window: int = 100

    # Mean Reversion
    mr_fast_window: int = 20
    mr_slow_window: int = 200

    # Sweep Detection
    sweep_time_window_ms: int = 100
    sweep_min_trades: int = 5
    sweep_price_threshold: float = 0.0005  # 0.05%

    # Z-score normalization
    zscore_window: int = 500


class OrderFlowFactorCalculator:
    """
    Calculator for Order Flow based factors.

    All factors are calculated on Dollar Bar data with the following columns:
    - start_time, end_time: Bar timestamps
    - open, high, low, close: OHLC prices
    - volume: Total volume
    - buy_volume, sell_volume: Directional volumes
    - dollar_volume: Total dollar amount
    - vwap: Volume weighted average price
    - tick_count: Number of trades in the bar

    Optional columns for enhanced factors:
    - buy_dollar, sell_dollar: Directional dollar amounts
    """

    def __init__(self, config: Optional[OrderFlowFactorConfig] = None):
        self.config = config or OrderFlowFactorConfig()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Order Flow factors.

        Args:
            df: Dollar Bar DataFrame with required columns

        Returns:
            DataFrame with all calculated factors added
        """
        result = df.copy()

        # Ensure required columns exist
        self._validate_columns(result)

        # Calculate buy/sell dollar if not present
        if "buy_dollar" not in result.columns:
            result["buy_dollar"] = result["buy_volume"] * result["vwap"]
        if "sell_dollar" not in result.columns:
            result["sell_dollar"] = result["sell_volume"] * result["vwap"]

        # Calculate each factor group
        result = self._calc_ofi(result)
        result = self._calc_smart_flow(result)
        result = self._calc_trade_intensity(result)
        result = self._calc_vpin(result)
        result = self._calc_kyle_lambda(result)
        result = self._calc_mr_signal(result)

        # Z-score normalize main factors
        result = self._calc_zscores(result)

        return result

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        required = ["close", "volume", "buy_volume", "sell_volume", "vwap"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _calc_ofi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Order Flow Imbalance at multiple time scales.

        OFI(n) = (Σ buy_dollar - Σ sell_dollar) / (Σ buy_dollar + Σ sell_dollar)

        - OFI > 0: Buy pressure dominates
        - OFI < 0: Sell pressure dominates
        """
        buy_d = df["buy_dollar"]
        sell_d = df["sell_dollar"]

        for window, name in [
            (self.config.ofi_fast_window, "OFI_fast"),
            (self.config.ofi_mid_window, "OFI_mid"),
            (self.config.ofi_slow_window, "OFI_slow"),
        ]:
            sum_buy = buy_d.rolling(window, min_periods=window // 2).sum()
            sum_sell = sell_d.rolling(window, min_periods=window // 2).sum()
            total = sum_buy + sum_sell
            df[name] = (sum_buy - sum_sell) / total.replace(0, np.nan)

        return df

    def _calc_smart_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Smart Flow (Large Order Flow) factor.

        Big trade threshold = rolling_mean(trade_dollar, 2000) * 5
        SmartFlow(n) = (Σ big_buy_dollar - Σ big_sell_dollar) / Σ big_trade_dollar

        Note: Since we're working with bars, we approximate by using
        bars with dollar_volume > threshold as "big trade bars".
        """
        # Calculate dynamic big trade threshold
        avg_dollar = df["dollar_volume"].rolling(
            self.config.big_trade_lookback, min_periods=100
        ).mean()
        big_threshold = avg_dollar * self.config.big_trade_multiplier

        # Identify big trade bars
        is_big = df["dollar_volume"] > big_threshold

        # Big buy/sell dollar (only for big trade bars)
        big_buy = df["buy_dollar"].where(is_big, 0)
        big_sell = df["sell_dollar"].where(is_big, 0)
        big_total = df["dollar_volume"].where(is_big, 0)

        for window in self.config.smart_flow_windows:
            sum_big_buy = big_buy.rolling(window, min_periods=window // 4).sum()
            sum_big_sell = big_sell.rolling(window, min_periods=window // 4).sum()
            sum_big_total = big_total.rolling(window, min_periods=window // 4).sum()

            df[f"SmartFlow_{window}"] = (sum_big_buy - sum_big_sell) / sum_big_total.replace(0, np.nan)

        # Also output the threshold for reference
        df["big_trade_threshold"] = big_threshold

        return df

    def _calc_trade_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Trade Intensity (λ) factor.

        λ(n) = num_trades[t-n:t] / Σ time_span[t-n:t]  (trades per second)
        λ_zscore = (λ - rolling_mean(λ, 500)) / rolling_std(λ, 500)

        High λ_zscore (> 2.0) indicates volume surge events.
        """
        # Time span per bar (in seconds)
        if "end_time" in df.columns and "start_time" in df.columns:
            time_span = (
                pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])
            ).dt.total_seconds().replace(0, np.nan)
        else:
            # Fallback: assume uniform time span
            time_span = pd.Series(1.0, index=df.index)

        # Trade count per bar
        if "tick_count" in df.columns:
            tick_count = df["tick_count"]
        elif "num_trades" in df.columns:
            tick_count = df["num_trades"]
        else:
            # Estimate from volume variance or use placeholder
            tick_count = pd.Series(1.0, index=df.index)

        # Rolling trade intensity
        window = self.config.lambda_window
        sum_trades = tick_count.rolling(window, min_periods=window // 2).sum()
        sum_time = time_span.rolling(window, min_periods=window // 2).sum()

        lambda_val = sum_trades / sum_time.replace(0, np.nan)
        df["lambda"] = lambda_val

        # Z-score normalization
        zscore_window = self.config.lambda_zscore_window
        roll_mean = lambda_val.rolling(zscore_window, min_periods=zscore_window // 4).mean()
        roll_std = lambda_val.rolling(zscore_window, min_periods=zscore_window // 4).std()

        df["lambda_zscore"] = (lambda_val - roll_mean) / roll_std.replace(0, np.nan)

        return df

    def _calc_vpin(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).

        VPIN(n) = mean(|buy_volume - sell_volume| / volume, past n bars)

        Higher VPIN indicates higher information asymmetry and expected volatility.
        Does not indicate direction, but can be used to adjust position sizing.
        """
        # Per-bar volume imbalance ratio
        imbalance = (df["buy_volume"] - df["sell_volume"]).abs() / df["volume"].replace(0, np.nan)

        window = self.config.vpin_window
        df["VPIN"] = imbalance.rolling(window, min_periods=window // 2).mean()

        return df

    def _calc_kyle_lambda(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Kyle Lambda (Price Impact) factor.

        Kyle_Lambda(n) = Σ|close[i] - close[i-1]| / Σ dollar_volume[i]  (past n bars)

        Higher Kyle Lambda = worse liquidity = reduce position size, widen stops
        Lower Kyle Lambda = better liquidity = can increase position size
        """
        abs_price_change = df["close"].diff().abs()

        window = self.config.kyle_lambda_window
        sum_price_change = abs_price_change.rolling(window, min_periods=window // 2).sum()
        sum_dollar = df["dollar_volume"].rolling(window, min_periods=window // 2).sum()

        df["Kyle_Lambda"] = sum_price_change / sum_dollar.replace(0, np.nan)

        # Also calculate percentile rank for condition checks
        df["Kyle_Lambda_pctrank"] = df["Kyle_Lambda"].rolling(
            self.config.lambda_zscore_window, min_periods=100
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        return df

    def _calc_mr_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Mean Reversion signal.

        MR(fast, slow) = (VWAP_rolling(fast) - VWAP_rolling(slow)) / realized_vol(slow)

        - MR > 0: Short-term price above long-term mean (potential short)
        - MR < 0: Short-term price below long-term mean (potential long)
        - |MR| > 2: Extreme deviation, higher reversion probability
        """
        fast = self.config.mr_fast_window
        slow = self.config.mr_slow_window

        vwap_fast = df["vwap"].rolling(fast, min_periods=fast // 2).mean()
        vwap_slow = df["vwap"].rolling(slow, min_periods=slow // 2).mean()

        # Realized volatility (standard deviation of returns)
        returns = df["close"].pct_change()
        realized_vol = returns.rolling(slow, min_periods=slow // 2).std()

        # Normalize by price to make MR scale-invariant
        price_diff = (vwap_fast - vwap_slow) / df["close"]
        df["MR"] = price_diff / realized_vol.replace(0, np.nan)

        return df

    def _calc_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-score normalized versions of main factors.

        Used for signal composition with standardized scales.
        """
        factors_to_zscore = [
            "OFI_fast", "OFI_mid", "OFI_slow",
            "SmartFlow_50", "SmartFlow_200",
            "VPIN", "Kyle_Lambda", "MR",
        ]

        window = self.config.zscore_window

        for factor in factors_to_zscore:
            if factor not in df.columns:
                continue

            vals = df[factor]
            roll_mean = vals.rolling(window, min_periods=window // 4).mean()
            roll_std = vals.rolling(window, min_periods=window // 4).std()

            df[f"{factor}_zscore"] = (vals - roll_mean) / roll_std.replace(0, np.nan)

        return df


def calculate_sweep_events(
    tick_df: pd.DataFrame,
    time_window_ms: int = 100,
    min_trades: int = 5,
    price_threshold: float = 0.0005,
    big_threshold_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Detect Sweep events from raw tick data.

    A Sweep is detected when within a time window:
    - Same direction consecutive trades >= min_trades
    - Cumulative amount > big_trade_threshold * multiplier
    - Price moves directionally > price_threshold

    Args:
        tick_df: Raw tick DataFrame with columns:
            - transact_time: Millisecond timestamp
            - price: Trade price
            - quantity: Trade quantity
            - is_buyer_maker: True if maker is buyer (taker is seller)
        time_window_ms: Time window in milliseconds
        min_trades: Minimum consecutive trades
        price_threshold: Minimum price movement ratio
        big_threshold_multiplier: Threshold multiplier

    Returns:
        DataFrame with sweep events:
            - timestamp: Event timestamp
            - direction: +1 (buy sweep) or -1 (sell sweep)
            - dollar_amount: Total dollar amount
            - price_change: Directional price change ratio
    """
    if len(tick_df) < min_trades:
        return pd.DataFrame(columns=["timestamp", "direction", "dollar_amount", "price_change"])

    # Convert is_buyer_maker to side: +1 = taker buy, -1 = taker sell
    # is_buyer_maker=True means taker is seller (-1)
    # is_buyer_maker=False means taker is buyer (+1)
    df = tick_df.copy()
    df["side"] = np.where(df["is_buyer_maker"], -1, 1)
    df["dollar_amount"] = df["price"] * df["quantity"]

    # Sort by time
    df = df.sort_values("transact_time").reset_index(drop=True)

    # Calculate rolling big trade threshold
    avg_dollar = df["dollar_amount"].rolling(2000, min_periods=100).mean()
    big_threshold = avg_dollar * big_threshold_multiplier

    sweeps = []
    i = 0
    n = len(df)

    while i < n - min_trades:
        start_time = df.iloc[i]["transact_time"]
        start_price = df.iloc[i]["price"]
        direction = df.iloc[i]["side"]

        # Look for consecutive same-direction trades within time window
        j = i + 1
        total_dollar = df.iloc[i]["dollar_amount"]
        same_direction_count = 1

        while j < n:
            row = df.iloc[j]
            time_diff = row["transact_time"] - start_time

            if time_diff > time_window_ms:
                break

            if row["side"] == direction:
                same_direction_count += 1
                total_dollar += row["dollar_amount"]
            else:
                # Direction changed, reset
                break

            j += 1

        end_price = df.iloc[j - 1]["price"]
        price_change = (end_price - start_price) / start_price * direction

        # Check sweep conditions
        threshold = big_threshold.iloc[i] if pd.notna(big_threshold.iloc[i]) else avg_dollar.mean() * big_threshold_multiplier

        if (same_direction_count >= min_trades and
            total_dollar > threshold and
            price_change > price_threshold):
            sweeps.append({
                "timestamp": start_time,
                "direction": direction,
                "dollar_amount": total_dollar,
                "price_change": price_change,
                "trade_count": same_direction_count,
            })
            i = j  # Skip processed trades
        else:
            i += 1

    return pd.DataFrame(sweeps)


def aggregate_sweeps_to_bars(
    sweep_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    time_col: str = "start_time",
) -> pd.DataFrame:
    """
    Aggregate sweep events to Dollar Bar level.

    For each bar, calculate:
    - sweep_count: Total number of sweeps
    - sweep_net_direction: Net direction (+1 = more buy sweeps, -1 = more sell)
    - sweep_buy_count: Number of buy sweeps
    - sweep_sell_count: Number of sell sweeps

    Args:
        sweep_df: Sweep events from calculate_sweep_events()
        bar_df: Dollar Bar DataFrame
        time_col: Time column name in bar_df

    Returns:
        bar_df with sweep columns added
    """
    result = bar_df.copy()

    # Initialize sweep columns
    result["sweep_count"] = 0
    result["sweep_net_direction"] = 0.0
    result["sweep_buy_count"] = 0
    result["sweep_sell_count"] = 0

    if len(sweep_df) == 0:
        return result

    # Get bar time boundaries
    bar_starts = pd.to_datetime(result[time_col])
    if "end_time" in result.columns:
        bar_ends = pd.to_datetime(result["end_time"])
    else:
        bar_ends = bar_starts.shift(-1)
        bar_ends.iloc[-1] = bar_starts.iloc[-1] + pd.Timedelta(hours=1)

    # Convert sweep timestamps to datetime
    sweep_times = pd.to_datetime(sweep_df["timestamp"], unit="ms", utc=True)

    # Assign sweeps to bars
    for idx in result.index:
        bar_start = bar_starts.loc[idx]
        bar_end = bar_ends.loc[idx]

        # Find sweeps in this bar's time range
        mask = (sweep_times >= bar_start) & (sweep_times < bar_end)
        bar_sweeps = sweep_df[mask]

        if len(bar_sweeps) > 0:
            result.loc[idx, "sweep_count"] = len(bar_sweeps)
            result.loc[idx, "sweep_buy_count"] = (bar_sweeps["direction"] == 1).sum()
            result.loc[idx, "sweep_sell_count"] = (bar_sweeps["direction"] == -1).sum()
            result.loc[idx, "sweep_net_direction"] = bar_sweeps["direction"].sum()

    return result


def calculate_order_flow_factors(
    bar_df: pd.DataFrame,
    config: Optional[OrderFlowFactorConfig] = None,
) -> pd.DataFrame:
    """
    Convenience function to calculate all Order Flow factors.

    Args:
        bar_df: Dollar Bar DataFrame
        config: Optional configuration

    Returns:
        DataFrame with all factors calculated
    """
    calculator = OrderFlowFactorCalculator(config)
    return calculator.calculate_all(bar_df)
