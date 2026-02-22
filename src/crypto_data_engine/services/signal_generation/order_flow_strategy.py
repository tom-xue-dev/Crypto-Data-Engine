"""
Order Flow Based Trading Strategies.

Implements strategies from the quantitative trading design:
1. OrderFlowMomentumStrategy: Multi-factor momentum based on order flow
2. MeanReversionStrategy: Mean reversion with order flow confirmation

Both strategies include:
- Signal composition with z-score normalized factors
- Entry conditions with multiple filters
- Exit conditions with stop-loss, take-profit, trailing stop, and time stop
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SignalType(Enum):
    """Trading signal type."""
    NONE = 0
    LONG = 1
    SHORT = -1
    EXIT_LONG = 2
    EXIT_SHORT = -2


class ExitReason(Enum):
    """Reason for position exit."""
    NONE = "none"
    SIGNAL_DECAY = "signal_decay"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    REVERSE_SWEEP = "reverse_sweep"
    MR_EXTREME = "mr_extreme"


@dataclass
class PositionState:
    """Track current position state."""
    direction: int = 0  # 0 = flat, 1 = long, -1 = short
    entry_price: float = 0.0
    entry_bar: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop_activated: bool = False


@dataclass
class OrderFlowMomentumConfig:
    """Configuration for Order Flow Momentum Strategy."""

    # Factor weights for composite score
    weight_ofi_fast: float = 0.30
    weight_ofi_mid: float = 0.25
    weight_smart_flow: float = 0.30
    weight_sweep: float = 0.15

    # Entry thresholds
    entry_score_threshold: float = 1.5  # |composite_score| > threshold to enter
    lambda_zscore_min: float = 0.5  # Trade intensity must be active
    vpin_max: float = 0.7  # Not too much information asymmetry
    sweep_lookback: int = 10  # Bars to check for reverse sweep

    # Exit thresholds
    exit_score_threshold: float = 0.5  # |composite_score| < threshold to exit
    take_profit_pct: float = 0.003  # 0.3% take profit
    stop_loss_pct: float = 0.0015  # 0.15% hard stop loss
    time_stop_bars: int = 200  # Max holding period

    # Trailing stop
    trailing_activate_pct: float = 0.001  # Activate at 0.1% profit
    trailing_breakeven_pct: float = 0.001  # First move stop to breakeven
    trailing_lock_pct: float = 0.002  # Lock 0.1% profit at 0.2% gain


@dataclass
class MeanReversionConfig:
    """Configuration for Mean Reversion Strategy."""

    # Entry thresholds
    mr_entry_threshold: float = 2.0  # |MR| > threshold to enter
    ofi_confirmation: bool = True  # Require OFI confirmation

    # Exit thresholds
    mr_exit_threshold: float = 0.5  # |MR| < threshold to exit
    mr_extreme_exit: float = 3.5  # |MR| > this means trend confirmed, exit
    stop_loss_pct: float = 0.002  # 0.2% hard stop loss
    time_stop_bars: int = 500  # Max holding period


class OrderFlowMomentumStrategy:
    """
    Order Flow Momentum Strategy.

    Signal composition:
        composite_score = w1*OFI_fast_zscore + w2*OFI_mid_zscore
                        + w3*SmartFlow_zscore + w4*Sweep_net_direction

    Entry conditions (LONG):
        1. composite_score > +1.5
        2. λ_zscore > 0.5 (active trading, not dead market)
        3. VPIN < 0.7 (not extreme information asymmetry)
        4. No reverse sweep in past 10 bars

    Exit conditions:
        - Signal decay: composite_score returns to ±0.5
        - Take profit: 0.3% gain
        - Stop loss: 0.15% loss (hard stop, cannot be breached)
        - Trailing stop: After 0.1% profit, move stop to breakeven
        - Time stop: 200 bars max holding
        - Reverse sweep: Opposite direction sweep event
    """

    def __init__(self, config: Optional[OrderFlowMomentumConfig] = None):
        self.config = config or OrderFlowMomentumConfig()
        self.position = PositionState()

    def calculate_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite score from order flow factors.

        Requires columns: OFI_fast_zscore, OFI_mid_zscore, SmartFlow_50_zscore, sweep_net_direction
        """
        cfg = self.config

        # Initialize score
        score = pd.Series(0.0, index=df.index)

        # OFI components
        if "OFI_fast_zscore" in df.columns:
            score += cfg.weight_ofi_fast * df["OFI_fast_zscore"].fillna(0)
        if "OFI_mid_zscore" in df.columns:
            score += cfg.weight_ofi_mid * df["OFI_mid_zscore"].fillna(0)

        # SmartFlow component
        if "SmartFlow_50_zscore" in df.columns:
            score += cfg.weight_smart_flow * df["SmartFlow_50_zscore"].fillna(0)
        elif "SmartFlow_200_zscore" in df.columns:
            score += cfg.weight_smart_flow * df["SmartFlow_200_zscore"].fillna(0)

        # Sweep component (already directional, just normalize by dividing by typical count)
        if "sweep_net_direction" in df.columns:
            sweep_norm = df["sweep_net_direction"] / (df["sweep_count"].replace(0, 1) + 1)
            score += cfg.weight_sweep * sweep_norm.fillna(0)

        return score

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the entire DataFrame.

        Args:
            df: DataFrame with order flow factors calculated

        Returns:
            DataFrame with columns:
            - composite_score: The combined signal score
            - signal: SignalType value
            - exit_reason: ExitReason value (if exiting)
        """
        result = df.copy()
        cfg = self.config

        # Calculate composite score
        result["composite_score"] = self.calculate_composite_score(df)

        # Initialize signal columns
        result["signal"] = SignalType.NONE.value
        result["exit_reason"] = ExitReason.NONE.value

        # Check for required columns
        has_lambda = "lambda_zscore" in df.columns
        has_vpin = "VPIN" in df.columns
        has_sweep = "sweep_net_direction" in df.columns

        # Reset position state
        self.position = PositionState()

        # Iterate through bars to generate signals
        for i in range(len(result)):
            bar_idx = result.index[i]
            score = result.loc[bar_idx, "composite_score"]

            # Get current price
            price = result.loc[bar_idx, "close"] if "close" in result.columns else 0

            # Check filters
            lambda_ok = (not has_lambda) or (result.loc[bar_idx, "lambda_zscore"] > cfg.lambda_zscore_min)
            vpin_ok = (not has_vpin) or (result.loc[bar_idx, "VPIN"] < cfg.vpin_max)

            # Check for reverse sweep in lookback
            reverse_sweep = False
            if has_sweep and self.position.direction != 0:
                lookback_start = max(0, i - cfg.sweep_lookback)
                recent_sweeps = result.iloc[lookback_start:i]["sweep_net_direction"]
                if self.position.direction == 1:  # Long position
                    reverse_sweep = (recent_sweeps < 0).any()
                elif self.position.direction == -1:  # Short position
                    reverse_sweep = (recent_sweeps > 0).any()

            # Process based on current position
            if self.position.direction == 0:
                # Flat - check for entry
                if score > cfg.entry_score_threshold and lambda_ok and vpin_ok:
                    # LONG entry
                    result.loc[bar_idx, "signal"] = SignalType.LONG.value
                    self.position = PositionState(
                        direction=1,
                        entry_price=price,
                        entry_bar=i,
                        highest_price=price,
                        lowest_price=price,
                    )
                elif score < -cfg.entry_score_threshold and lambda_ok and vpin_ok:
                    # SHORT entry
                    result.loc[bar_idx, "signal"] = SignalType.SHORT.value
                    self.position = PositionState(
                        direction=-1,
                        entry_price=price,
                        entry_bar=i,
                        highest_price=price,
                        lowest_price=price,
                    )

            else:
                # In position - check for exit
                bars_held = i - self.position.entry_bar
                pnl_pct = (price - self.position.entry_price) / self.position.entry_price * self.position.direction

                # Update high/low water marks
                if price > self.position.highest_price:
                    self.position.highest_price = price
                if price < self.position.lowest_price:
                    self.position.lowest_price = price

                exit_signal = False
                exit_reason = ExitReason.NONE

                # Check exit conditions in priority order

                # 1. Hard stop loss (highest priority, cannot be breached)
                if pnl_pct < -cfg.stop_loss_pct:
                    exit_signal = True
                    exit_reason = ExitReason.STOP_LOSS

                # 2. Reverse sweep
                elif reverse_sweep:
                    exit_signal = True
                    exit_reason = ExitReason.REVERSE_SWEEP

                # 3. Take profit
                elif pnl_pct >= cfg.take_profit_pct:
                    exit_signal = True
                    exit_reason = ExitReason.TAKE_PROFIT

                # 4. Trailing stop
                elif self.position.trailing_stop_activated:
                    # Calculate trailing stop level
                    if pnl_pct >= cfg.trailing_lock_pct:
                        stop_level = cfg.trailing_breakeven_pct
                    else:
                        stop_level = 0.0  # Breakeven

                    if pnl_pct < stop_level:
                        exit_signal = True
                        exit_reason = ExitReason.TRAILING_STOP

                # 5. Activate trailing stop
                elif pnl_pct >= cfg.trailing_activate_pct:
                    self.position.trailing_stop_activated = True

                # 6. Signal decay
                if not exit_signal:
                    if self.position.direction == 1 and score < cfg.exit_score_threshold:
                        exit_signal = True
                        exit_reason = ExitReason.SIGNAL_DECAY
                    elif self.position.direction == -1 and score > -cfg.exit_score_threshold:
                        exit_signal = True
                        exit_reason = ExitReason.SIGNAL_DECAY

                # 7. Time stop
                if not exit_signal and bars_held >= cfg.time_stop_bars:
                    exit_signal = True
                    exit_reason = ExitReason.TIME_STOP

                # Execute exit
                if exit_signal:
                    if self.position.direction == 1:
                        result.loc[bar_idx, "signal"] = SignalType.EXIT_LONG.value
                    else:
                        result.loc[bar_idx, "signal"] = SignalType.EXIT_SHORT.value
                    result.loc[bar_idx, "exit_reason"] = exit_reason.value
                    self.position = PositionState()

        return result


class MeanReversionStrategy:
    """
    Mean Reversion Strategy with Order Flow Confirmation.

    Entry conditions (LONG):
        1. MR < -2.0 (price significantly below mean)
        2. OFI_fast > 0 (short-term buying confirms reversal)
        3. Kyle_Lambda < 50th percentile (liquidity is acceptable)

    Exit conditions:
        - Mean reversion: MR returns to ±0.5
        - Stop loss: 0.2% loss
        - MR extreme: |MR| > 3.5 (trend confirmed, exit)
        - Time stop: 500 bars max holding
    """

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.config = config or MeanReversionConfig()
        self.position = PositionState()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the entire DataFrame.

        Args:
            df: DataFrame with MR, OFI_fast, Kyle_Lambda_pctrank columns

        Returns:
            DataFrame with signal and exit_reason columns
        """
        result = df.copy()
        cfg = self.config

        # Initialize signal columns
        result["signal"] = SignalType.NONE.value
        result["exit_reason"] = ExitReason.NONE.value

        # Check for required columns
        has_mr = "MR" in df.columns
        has_ofi = "OFI_fast" in df.columns
        has_kyle = "Kyle_Lambda_pctrank" in df.columns

        if not has_mr:
            return result

        # Reset position state
        self.position = PositionState()

        # Iterate through bars
        for i in range(len(result)):
            bar_idx = result.index[i]
            mr = result.loc[bar_idx, "MR"]
            price = result.loc[bar_idx, "close"] if "close" in result.columns else 0

            # Get filter values
            ofi = result.loc[bar_idx, "OFI_fast"] if has_ofi else 0
            kyle_pct = result.loc[bar_idx, "Kyle_Lambda_pctrank"] if has_kyle else 0.5

            if self.position.direction == 0:
                # Flat - check for entry
                kyle_ok = (not has_kyle) or (kyle_pct < 0.5)

                if mr < -cfg.mr_entry_threshold and kyle_ok and (not cfg.ofi_confirmation or ofi > 0):
                    # LONG entry (price below mean, expect reversion up)
                    result.loc[bar_idx, "signal"] = SignalType.LONG.value
                    self.position = PositionState(
                        direction=1,
                        entry_price=price,
                        entry_bar=i,
                        highest_price=price,
                        lowest_price=price,
                    )
                elif mr > cfg.mr_entry_threshold and kyle_ok and (not cfg.ofi_confirmation or ofi < 0):
                    # SHORT entry (price above mean, expect reversion down)
                    result.loc[bar_idx, "signal"] = SignalType.SHORT.value
                    self.position = PositionState(
                        direction=-1,
                        entry_price=price,
                        entry_bar=i,
                        highest_price=price,
                        lowest_price=price,
                    )

            else:
                # In position - check for exit
                bars_held = i - self.position.entry_bar
                pnl_pct = (price - self.position.entry_price) / self.position.entry_price * self.position.direction

                exit_signal = False
                exit_reason = ExitReason.NONE

                # 1. Hard stop loss
                if pnl_pct < -cfg.stop_loss_pct:
                    exit_signal = True
                    exit_reason = ExitReason.STOP_LOSS

                # 2. MR extreme (trend confirmed, cut loss)
                elif self.position.direction == 1 and mr < -cfg.mr_extreme_exit:
                    exit_signal = True
                    exit_reason = ExitReason.MR_EXTREME
                elif self.position.direction == -1 and mr > cfg.mr_extreme_exit:
                    exit_signal = True
                    exit_reason = ExitReason.MR_EXTREME

                # 3. Mean reversion achieved
                elif abs(mr) < cfg.mr_exit_threshold:
                    exit_signal = True
                    exit_reason = ExitReason.SIGNAL_DECAY

                # 4. Time stop
                elif bars_held >= cfg.time_stop_bars:
                    exit_signal = True
                    exit_reason = ExitReason.TIME_STOP

                # Execute exit
                if exit_signal:
                    if self.position.direction == 1:
                        result.loc[bar_idx, "signal"] = SignalType.EXIT_LONG.value
                    else:
                        result.loc[bar_idx, "signal"] = SignalType.EXIT_SHORT.value
                    result.loc[bar_idx, "exit_reason"] = exit_reason.value
                    self.position = PositionState()

        return result


@dataclass
class SignalSummary:
    """Summary statistics for generated signals."""
    total_bars: int = 0
    long_entries: int = 0
    short_entries: int = 0
    long_exits: int = 0
    short_exits: int = 0
    exit_reasons: Dict[str, int] = field(default_factory=dict)


def summarize_signals(df: pd.DataFrame) -> SignalSummary:
    """Summarize signal statistics from a signal DataFrame."""
    summary = SignalSummary(total_bars=len(df))

    if "signal" in df.columns:
        summary.long_entries = (df["signal"] == SignalType.LONG.value).sum()
        summary.short_entries = (df["signal"] == SignalType.SHORT.value).sum()
        summary.long_exits = (df["signal"] == SignalType.EXIT_LONG.value).sum()
        summary.short_exits = (df["signal"] == SignalType.EXIT_SHORT.value).sum()

    if "exit_reason" in df.columns:
        for reason in ExitReason:
            if reason != ExitReason.NONE:
                count = (df["exit_reason"] == reason.value).sum()
                if count > 0:
                    summary.exit_reasons[reason.value] = count

    return summary
