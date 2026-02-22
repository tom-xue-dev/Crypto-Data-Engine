"""
Tick Feature Enricher — add tick microstructure columns to dollar bars.

For each bar, computes 6 tick-level factors (→ 9 columns) using a rolling
window of raw aggTrades ticks aligned to the bar's volume-clock boundaries.

Usage:
    from crypto_data_engine.services.bar_aggregator.tick_feature_enricher import (
        TickFeatureEnricher, TickFeatureEnricherConfig,
    )
    enricher = TickFeatureEnricher()
    enriched = enricher.enrich(bars_df, tick_df)

Integration:
    Called by `enrich_cmd.py` CLI as a post-processing step on existing bars.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from crypto_data_engine.services.feature.tick_microstructure_factors import (
    TickMicrostructureConfig,
    compute_burstiness,
    compute_jump_ratio,
    compute_kyle_lambda,
    compute_toxicity,
    compute_vpin,
    compute_whale_metrics,
)

logger = logging.getLogger(__name__)

# Columns added by the enricher
TICK_FEATURE_COLUMNS = [
    "tick_vpin",
    "tick_toxicity_run_mean",
    "tick_toxicity_run_max",
    "tick_toxicity_ratio",
    "tick_kyle_lambda",
    "tick_burstiness",
    "tick_jump_ratio",
    "tick_whale_imbalance",
    "tick_whale_impact",
]


@dataclass
class TickFeatureEnricherConfig:
    """Configuration for tick feature enrichment."""

    lookback_bars: int = 5
    """Number of preceding bars to include in the rolling tick window."""

    min_ticks: int = 500
    """Minimum ticks in window required to compute features; else NaN."""

    micro_config: TickMicrostructureConfig = None  # type: ignore[assignment]
    """Underlying factor computation config."""

    def __post_init__(self):
        if self.micro_config is None:
            self.micro_config = TickMicrostructureConfig()


class TickFeatureEnricher:
    """Enrich dollar bars with tick-level microstructure features."""

    def __init__(self, config: Optional[TickFeatureEnricherConfig] = None):
        self.cfg = config or TickFeatureEnricherConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        bars: pd.DataFrame,
        ticks: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add 9 tick microstructure columns to *bars*.

        Args:
            bars:  DataFrame with at least ``start_time`` and ``end_time``
                   columns (datetime64, UTC-aware).
            ticks: Normalised tick DataFrame with columns
                   ``timestamp`` (int64 ms), ``price`` (float64),
                   ``quantity`` (float64), ``is_buyer_maker`` (bool).

        Returns:
            A copy of *bars* with 9 additional ``tick_*`` columns.
        """
        n_bars = len(bars)
        if n_bars == 0 or len(ticks) == 0:
            return self._empty_result(bars)

        # --- prepare tick arrays (assume sorted by timestamp; verify cheaply) ---
        ts_raw = ticks["timestamp"].values
        if len(ts_raw) > 1 and ts_raw[0] > ts_raw[-1]:
            sort_idx = np.argsort(ts_raw)
            ts_arr = ts_raw[sort_idx].astype(np.int64)
            price_arr = ticks["price"].values[sort_idx].astype(np.float64)
            qty_arr = ticks["quantity"].values[sort_idx].astype(np.float64)
            ibm_arr = ticks["is_buyer_maker"].values[sort_idx].astype(bool)
        else:
            ts_arr = ts_raw.astype(np.int64)
            price_arr = ticks["price"].values.astype(np.float64)
            qty_arr = ticks["quantity"].values.astype(np.float64)
            ibm_arr = ticks["is_buyer_maker"].values.astype(bool)

        # --- bar time boundaries (ms epoch) ---
        bar_starts_ms = self._to_epoch_ms(bars["start_time"].values)
        bar_ends_ms = self._to_epoch_ms(bars["end_time"].values)

        # --- pre-compute ALL per-bar tick index ranges via vectorized searchsorted ---
        lookback = self.cfg.lookback_bars
        min_ticks = self.cfg.min_ticks
        mc = self.cfg.micro_config

        lo_bar_indices = np.maximum(0, np.arange(n_bars) - lookback + 1)
        all_t_lo = bar_starts_ms[lo_bar_indices]
        all_t_hi = bar_ends_ms  # already length n_bars

        all_idx_lo = np.searchsorted(ts_arr, all_t_lo, side="left")
        all_idx_hi = np.searchsorted(ts_arr, all_t_hi, side="right")

        # Pre-allocate result arrays
        res = {col: np.full(n_bars, np.nan) for col in TICK_FEATURE_COLUMNS}

        for i in range(n_bars):
            idx_lo = all_idx_lo[i]
            idx_hi = all_idx_hi[i]

            n_ticks = idx_hi - idx_lo
            if n_ticks < min_ticks:
                continue

            # Slice tick arrays for this window
            w_ts = ts_arr[idx_lo:idx_hi]
            w_price = price_arr[idx_lo:idx_hi]
            w_qty = qty_arr[idx_lo:idx_hi]
            w_ibm = ibm_arr[idx_lo:idx_hi]

            # Compute all 6 factors
            res["tick_vpin"][i] = compute_vpin(
                w_qty, w_ibm, mc.vpin_n_buckets
            )

            rm, rx, tr = compute_toxicity(w_ibm)
            res["tick_toxicity_run_mean"][i] = rm
            res["tick_toxicity_run_max"][i] = rx
            res["tick_toxicity_ratio"][i] = tr

            res["tick_kyle_lambda"][i] = compute_kyle_lambda(
                w_price, w_qty, w_ibm, w_ts,
                mc.kyle_freq_ms, mc.kyle_min_buckets,
            )

            res["tick_burstiness"][i] = compute_burstiness(w_ts)

            res["tick_jump_ratio"][i] = compute_jump_ratio(
                w_price, w_ts, mc.jump_freq_ms, mc.jump_min_intervals,
            )

            wi, wp = compute_whale_metrics(
                w_price, w_qty, w_ibm, w_ts,
                mc.whale_percentile, mc.whale_impact_delay_ms,
            )
            res["tick_whale_imbalance"][i] = wi
            res["tick_whale_impact"][i] = wp

        # Build result DataFrame
        enriched = bars.copy()
        for col in TICK_FEATURE_COLUMNS:
            enriched[col] = res[col]
        return enriched

    # ------------------------------------------------------------------
    # File-level convenience (for multiprocessing)
    # ------------------------------------------------------------------

    def enrich_file_pair(
        self,
        bar_path: str,
        tick_path: str,
        output_path: str,
    ) -> Dict:
        """
        Enrich one bar parquet using its matching tick parquet, save result.

        Returns:
            Dict with keys: bar_path, n_bars, n_ticks, n_enriched, status.
        """
        from crypto_data_engine.services.bar_aggregator.tick_normalizer import (
            normalize_tick_data,
        )

        try:
            bars = pd.read_parquet(bar_path)
            raw_ticks = pd.read_parquet(tick_path)
            ticks = normalize_tick_data(raw_ticks, source_hint=Path(tick_path).name)
        except Exception as e:
            return dict(
                bar_path=bar_path, n_bars=0, n_ticks=0,
                n_enriched=0, status=f"load_error: {e}",
            )

        enriched = self.enrich(bars, ticks)
        n_enriched = int(enriched["tick_vpin"].notna().sum())

        enriched.to_parquet(output_path, index=False)

        return dict(
            bar_path=bar_path,
            n_bars=len(bars),
            n_ticks=len(ticks) if ticks is not None else 0,
            n_enriched=n_enriched,
            status="ok",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_epoch_ms(dt_arr) -> np.ndarray:
        """Convert datetime64 array (any resolution) to int64 milliseconds."""
        if isinstance(dt_arr, pd.DatetimeIndex):
            # asi8 returns int64 in the index's resolution (ms, us, or ns)
            resolution = getattr(dt_arr.dtype, "unit", "ns")
            raw = dt_arr.asi8
            if resolution == "ns":
                return raw // 1_000_000
            elif resolution == "us":
                return raw // 1_000
            elif resolution == "ms":
                return raw
            elif resolution == "s":
                return raw * 1_000
            return raw // 1_000_000  # fallback assume ns

        # pd.arrays.DatetimeArray or similar with asi8
        if hasattr(dt_arr, "asi8"):
            dtype = getattr(dt_arr, "dtype", None)
            unit = getattr(dtype, "unit", "ns")
            raw = np.asarray(dt_arr.asi8)
            if unit == "ms":
                return raw
            elif unit == "us":
                return raw // 1_000
            elif unit == "s":
                return raw * 1_000
            return raw // 1_000_000

        arr = np.asarray(dt_arr)
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ms]").astype(np.int64)

        # Already numeric
        return arr.astype(np.int64)

    @staticmethod
    def _empty_result(bars: pd.DataFrame) -> pd.DataFrame:
        """Return bars with NaN tick feature columns."""
        out = bars.copy()
        for col in TICK_FEATURE_COLUMNS:
            out[col] = np.nan
        return out


# ======================================================================
# Module-level convenience for multiprocessing workers
# ======================================================================

def enrich_file_pair_worker(args: Tuple[str, str, str, dict]) -> Dict:
    """
    Multiprocessing-safe worker function.

    Args:
        args: (bar_path, tick_path, output_path, config_dict)
    """
    bar_path, tick_path, output_path, cfg_dict = args
    config = TickFeatureEnricherConfig(**cfg_dict) if cfg_dict else None
    enricher = TickFeatureEnricher(config)
    return enricher.enrich_file_pair(bar_path, tick_path, output_path)
