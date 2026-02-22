"""
Hybrid momentum strategy combining slow cross-sectional pool selection
with fast per-bar signal generation.

Slow layer (weekly/monthly):
    - Rank all symbols by composite factor score
    - Select top N as long pool, bottom N as short pool
    - Neutral symbols are not traded

Fast layer (per-bar):
    - For pool-eligible symbols, generate entry/exit signals
    - Uses momentum, volume, and microstructure features
    - TP/SL/time-stop for risk management
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from crypto_data_engine.core.base import BaseStrategy, SignalType


@dataclass
class HybridStrategyConfig:
    """Configuration for hybrid momentum strategy."""

    # --- Slow layer (cross-sectional pool selection) ---
    pool_size: int = 30  # top N long + bottom N short
    rebalance_freq: str = "W"  # W=weekly, MS=monthly-start

    # --- Fast layer signal thresholds ---
    entry_threshold: float = 2.0  # composite z-score to enter
    exit_threshold: float = -0.5  # exit when score reverses to this level

    # --- Risk / TP / SL (adapted for K=50, per-bar std ≈ 0.3%) ---
    take_profit_pct: float = 0.015  # 1.5% ≈ 5σ per bar
    stop_loss_pct: float = 0.008  # 0.8% ≈ 2.7σ per bar
    time_stop_bars: int = 150  # ~3 days at K=50

    # --- Trade frequency control ---
    min_holding_bars: int = 20  # minimum bars before allowing exit (~10h)
    cooldown_bars: int = 10  # bars to wait after exit before re-entry (~5h)

    # --- Factor weights for fast-layer composite signal ---
    w_momentum: float = 0.30
    w_volume_surge: float = 0.20
    w_imbalance: float = 0.20
    w_vwap_dev: float = 0.15
    w_path_eff: float = 0.15


class HybridMomentumStrategy(BaseStrategy):
    """
    Hybrid strategy: slow cross-sectional selection + fast per-bar signals.

    Expects bar data to contain pre-computed columns (added by preprocessor):
        - pool_direction : int   (1=long pool, -1=short pool, 0=neutral)
        - cs_rank        : float (0-1 cross-sectional percentile rank)
        - momentum_5     : float (5-bar return)
        - momentum_20    : float (20-bar return)
        - vol_20         : float (20-bar rolling std of returns)
        - volume_ratio   : float (current volume / 20-bar avg volume)
        - rsi_14         : float (14-bar RSI)
        + raw bar columns: buy_sell_imbalance, vwap, close, path_efficiency, etc.
    """

    def __init__(self, config: Optional[HybridStrategyConfig] = None):
        super().__init__(name="HybridMomentum")
        self.config = config or HybridStrategyConfig()
        self._bars_held: Dict[str, int] = {}  # asset -> bars since entry
        self._cooldown: Dict[str, int] = {}  # asset -> bars since last exit

    def generate_signal(
        self, bar_data: pd.Series, position: Optional[float] = None
    ) -> SignalType:
        pool = bar_data.get("pool_direction", 0)
        has_position = position is not None and abs(position) > 0
        is_long = position is not None and position > 0

        # Pure cross-sectional momentum (long-only):
        # Hold long pool symbols, close everything else.
        # Engine has allow_short=False so SELL only closes, never opens short.

        if pool == 1:
            if not has_position:
                return SignalType.BUY
            return SignalType.HOLD
        else:
            # Not in long pool: close if holding
            if has_position and is_long:
                return SignalType.SELL
            return SignalType.HOLD

    # ------------------------------------------------------------------
    # Fast signal components
    # ------------------------------------------------------------------

    def _compute_fast_score(self, bar: pd.Series) -> float:
        """
        Compute composite z-score from multiple fast-layer features.
        Positive score = bullish, negative = bearish.
        """
        cfg = self.config
        components = []
        weights = []

        # 1. Short-term momentum z-score
        mom5 = bar.get("momentum_5", np.nan)
        vol20 = bar.get("vol_20", np.nan)
        if pd.notna(mom5) and pd.notna(vol20) and vol20 > 0:
            mom_z = mom5 / vol20
            components.append(np.clip(mom_z, -4, 4))
            weights.append(cfg.w_momentum)

        # 2. Volume surge (current bar volume vs 20-bar avg)
        vol_ratio = bar.get("volume_ratio", np.nan)
        if pd.notna(vol_ratio) and vol_ratio > 0:
            vol_z = np.log(vol_ratio)  # log scale: 1.0 → 0, 2.0 → 0.69
            components.append(np.clip(vol_z, -3, 3))
            weights.append(cfg.w_volume_surge)

        # 3. Buy/sell imbalance (already in [-1, 1])
        imbalance = bar.get("buy_sell_imbalance", np.nan)
        if pd.notna(imbalance):
            components.append(np.clip(imbalance * 3, -3, 3))
            weights.append(cfg.w_imbalance)

        # 4. VWAP deviation: close above VWAP = bullish
        close = bar.get("close", np.nan)
        vwap = bar.get("vwap", np.nan)
        if pd.notna(close) and pd.notna(vwap) and vwap > 0 and pd.notna(vol20) and vol20 > 0:
            vwap_dev = (close - vwap) / vwap / vol20
            components.append(np.clip(vwap_dev, -4, 4))
            weights.append(cfg.w_vwap_dev)

        # 5. Path efficiency (high = trending, directional)
        path_eff = bar.get("path_efficiency", np.nan)
        path_eff_ma = bar.get("path_efficiency_ma", np.nan)
        if pd.notna(path_eff) and pd.notna(path_eff_ma) and path_eff_ma > 0:
            pe_ratio = (path_eff / path_eff_ma) - 1.0
            mom5_val = bar.get("momentum_5", 0)
            direction = np.sign(mom5_val) if pd.notna(mom5_val) else 0
            components.append(np.clip(pe_ratio * direction * 3, -3, 3))
            weights.append(cfg.w_path_eff)

        if not components:
            return 0.0

        weights = np.array(weights)
        weights /= weights.sum()
        return float(np.dot(components, weights))
