"""
High-performance bar aggregation using Numba and multiprocessing.

Provides significant speedup over pure pandas implementation:
- Numba JIT compilation for inner loops
- Multiprocessing for batch file processing
- Memory-efficient streaming for large datasets
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Try to import numba, fallback to pure python if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from .bar_types import BarConfig, BarType, BaseBarBuilder, get_bar_builder

# Interval string → millisecond lookup used by the Numba time-bar kernel.
_INTERVAL_MS_MAP: Dict[str, int] = {
    "1s": 1_000, "5s": 5_000, "10s": 10_000, "15s": 15_000, "30s": 30_000,
    "1min": 60_000, "5min": 300_000, "10min": 600_000, "15min": 900_000,
    "30min": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1D": 86_400_000,
}


def _interval_to_ms(interval: str) -> int:
    """Convert a human-readable interval string to milliseconds."""
    if interval in _INTERVAL_MS_MAP:
        return _INTERVAL_MS_MAP[interval]
    # Fallback: try pandas Timedelta
    return int(pd.Timedelta(interval).total_seconds() * 1_000)


# ============================================================================
# Numba-accelerated functions
# ============================================================================

@jit(nopython=True, cache=True)
def _aggregate_time_bars_numba(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps: np.ndarray,
    is_buyer_maker: np.ndarray,
    interval_ms: int,
    include_advanced: bool,
) -> Tuple:
    """Single-pass Numba kernel for time-bar aggregation.

    Groups ticks by ``timestamp // interval_ms`` and accumulates all
    bar-level statistics in one traversal.  Returns parallel arrays of
    bar fields that the caller converts into a DataFrame.

    Returns a tuple of numpy arrays – one per bar field:
        (start_times, end_times, opens, highs, lows, closes,
         volumes, buy_volumes, sell_volumes, vwaps, tick_counts,
         dollar_volumes,
         price_stds, volume_stds, up_ratios, down_ratios, reversals,
         imbalances, max_vols, max_ratios, interval_means,
         path_efficiencies, impact_densities)
    """
    n = len(prices)
    # Estimate upper bound for number of bars
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return (empty, empty, empty, empty, empty, empty,
                empty, empty, empty, empty, empty_i, empty,
                empty, empty, empty, empty, empty_i,
                empty, empty, empty, empty,
                empty, empty)

    # Tight upper-bound estimate based on the time span, not tick count.
    # This avoids allocating gigabytes of output arrays for large tick datasets.
    time_span = timestamps[n - 1] - timestamps[0]
    estimated_bars = int(time_span / interval_ms) + 2 if interval_ms > 0 else n
    # Safety cap: never exceed tick count (e.g. if interval is tiny)
    max_bars = min(estimated_bars, n)
    start_times = np.empty(max_bars, dtype=np.float64)
    end_times = np.empty(max_bars, dtype=np.float64)
    opens = np.empty(max_bars, dtype=np.float64)
    highs = np.empty(max_bars, dtype=np.float64)
    lows = np.empty(max_bars, dtype=np.float64)
    closes = np.empty(max_bars, dtype=np.float64)
    volumes = np.empty(max_bars, dtype=np.float64)
    buy_vols = np.empty(max_bars, dtype=np.float64)
    sell_vols = np.empty(max_bars, dtype=np.float64)
    vwaps = np.empty(max_bars, dtype=np.float64)
    tick_counts = np.empty(max_bars, dtype=np.int64)
    dollar_vols = np.empty(max_bars, dtype=np.float64)
    # Advanced stats
    price_stds_out = np.empty(max_bars, dtype=np.float64)
    volume_stds_out = np.empty(max_bars, dtype=np.float64)
    up_ratios_out = np.empty(max_bars, dtype=np.float64)
    down_ratios_out = np.empty(max_bars, dtype=np.float64)
    reversals_out = np.empty(max_bars, dtype=np.int64)
    imbalances_out = np.empty(max_bars, dtype=np.float64)
    max_vols_out = np.empty(max_bars, dtype=np.float64)
    max_ratios_out = np.empty(max_bars, dtype=np.float64)
    interval_means_out = np.empty(max_bars, dtype=np.float64)
    path_eff_out = np.empty(max_bars, dtype=np.float64)
    impact_den_out = np.empty(max_bars, dtype=np.float64)

    bar_count = 0
    current_bar_id = timestamps[0] // interval_ms

    # Running accumulators for the current bar
    bar_start_ts = timestamps[0]
    bar_end_ts = timestamps[0]
    bar_open = prices[0]
    bar_high = prices[0]
    bar_low = prices[0]
    bar_close = prices[0]
    bar_volume = 0.0
    bar_buy_vol = 0.0
    bar_sell_vol = 0.0
    bar_dollar_vol = 0.0
    bar_tick_count = 0
    # Advanced accumulators
    bar_price_sum = 0.0
    bar_price_sq_sum = 0.0
    bar_vol_sum = 0.0
    bar_vol_sq_sum = 0.0
    bar_up_moves = 0
    bar_down_moves = 0
    bar_prev_direction = 0
    bar_reversals = 0
    bar_max_vol = 0.0
    bar_path_length = 0.0
    bar_interval_sum = 0.0
    bar_interval_count = 0
    bar_prev_price = prices[0]

    for i in range(n):
        tick_bar_id = timestamps[i] // interval_ms
        p = prices[i]
        q = quantities[i]

        if tick_bar_id != current_bar_id:
            # ---- Flush current bar ----
            vwap_val = bar_dollar_vol / bar_volume if bar_volume > 0.0 else bar_close
            total_moves = bar_up_moves + bar_down_moves

            start_times[bar_count] = bar_start_ts
            end_times[bar_count] = bar_end_ts
            opens[bar_count] = bar_open
            highs[bar_count] = bar_high
            lows[bar_count] = bar_low
            closes[bar_count] = bar_close
            volumes[bar_count] = bar_volume
            buy_vols[bar_count] = bar_buy_vol
            sell_vols[bar_count] = bar_sell_vol
            vwaps[bar_count] = vwap_val
            tick_counts[bar_count] = bar_tick_count
            dollar_vols[bar_count] = bar_dollar_vol

            if include_advanced:
                pmean = bar_price_sum / bar_tick_count if bar_tick_count > 0 else 0.0
                pvar = bar_price_sq_sum / bar_tick_count - pmean * pmean if bar_tick_count > 0 else 0.0
                price_stds_out[bar_count] = np.sqrt(pvar) if pvar > 0.0 else 0.0
                vmean = bar_vol_sum / bar_tick_count if bar_tick_count > 0 else 0.0
                vvar = bar_vol_sq_sum / bar_tick_count - vmean * vmean if bar_tick_count > 0 else 0.0
                volume_stds_out[bar_count] = np.sqrt(vvar) if vvar > 0.0 else 0.0
                up_ratios_out[bar_count] = bar_up_moves / total_moves if total_moves > 0 else 0.5
                down_ratios_out[bar_count] = bar_down_moves / total_moves if total_moves > 0 else 0.5
                reversals_out[bar_count] = bar_reversals
                imbalances_out[bar_count] = (bar_buy_vol - bar_sell_vol) / bar_volume if bar_volume > 0.0 else 0.0
                max_vols_out[bar_count] = bar_max_vol
                max_ratios_out[bar_count] = bar_max_vol / bar_volume if bar_volume > 0.0 else 0.0
                interval_means_out[bar_count] = bar_interval_sum / bar_interval_count if bar_interval_count > 0 else 0.0
                net_move = bar_close - bar_open
                abs_net = net_move if net_move >= 0 else -net_move
                path_eff_out[bar_count] = abs_net / bar_path_length if bar_path_length > 0.0 else 0.0
                impact_den_out[bar_count] = abs_net / bar_dollar_vol if bar_dollar_vol > 0.0 else 0.0

            bar_count += 1

            # ---- Reset accumulators for new bar ----
            current_bar_id = tick_bar_id
            bar_start_ts = timestamps[i]
            bar_open = p
            bar_high = p
            bar_low = p
            bar_close = p
            bar_volume = 0.0
            bar_buy_vol = 0.0
            bar_sell_vol = 0.0
            bar_dollar_vol = 0.0
            bar_tick_count = 0
            bar_price_sum = 0.0
            bar_price_sq_sum = 0.0
            bar_vol_sum = 0.0
            bar_vol_sq_sum = 0.0
            bar_up_moves = 0
            bar_down_moves = 0
            bar_prev_direction = 0
            bar_reversals = 0
            bar_max_vol = 0.0
            bar_path_length = 0.0
            bar_interval_sum = 0.0
            bar_interval_count = 0
            bar_prev_price = p

        # ---- Accumulate current tick into bar ----
        bar_end_ts = timestamps[i]
        bar_close = p
        if p > bar_high:
            bar_high = p
        if p < bar_low:
            bar_low = p
        bar_volume += q
        bar_dollar_vol += p * q
        bar_tick_count += 1

        if is_buyer_maker[i]:
            bar_sell_vol += q
        else:
            bar_buy_vol += q

        if include_advanced:
            bar_price_sum += p
            bar_price_sq_sum += p * p
            bar_vol_sum += q
            bar_vol_sq_sum += q * q
            if q > bar_max_vol:
                bar_max_vol = q

            if bar_tick_count > 1:
                diff = p - bar_prev_price
                abs_diff = diff if diff >= 0 else -diff
                bar_path_length += abs_diff
                if diff > 0:
                    bar_up_moves += 1
                    direction = 1
                elif diff < 0:
                    bar_down_moves += 1
                    direction = -1
                else:
                    direction = bar_prev_direction
                if direction != 0 and bar_prev_direction != 0 and direction != bar_prev_direction:
                    bar_reversals += 1
                bar_prev_direction = direction

                interval_val = timestamps[i] - timestamps[i - 1]
                bar_interval_sum += interval_val
                bar_interval_count += 1

            bar_prev_price = p

    # ---- Flush the last bar ----
    if bar_tick_count > 0:
        vwap_val = bar_dollar_vol / bar_volume if bar_volume > 0.0 else bar_close
        total_moves = bar_up_moves + bar_down_moves

        start_times[bar_count] = bar_start_ts
        end_times[bar_count] = bar_end_ts
        opens[bar_count] = bar_open
        highs[bar_count] = bar_high
        lows[bar_count] = bar_low
        closes[bar_count] = bar_close
        volumes[bar_count] = bar_volume
        buy_vols[bar_count] = bar_buy_vol
        sell_vols[bar_count] = bar_sell_vol
        vwaps[bar_count] = vwap_val
        tick_counts[bar_count] = bar_tick_count
        dollar_vols[bar_count] = bar_dollar_vol

        if include_advanced:
            pmean = bar_price_sum / bar_tick_count if bar_tick_count > 0 else 0.0
            pvar = bar_price_sq_sum / bar_tick_count - pmean * pmean if bar_tick_count > 0 else 0.0
            price_stds_out[bar_count] = np.sqrt(pvar) if pvar > 0.0 else 0.0
            vmean = bar_vol_sum / bar_tick_count if bar_tick_count > 0 else 0.0
            vvar = bar_vol_sq_sum / bar_tick_count - vmean * vmean if bar_tick_count > 0 else 0.0
            volume_stds_out[bar_count] = np.sqrt(vvar) if vvar > 0.0 else 0.0
            up_ratios_out[bar_count] = bar_up_moves / total_moves if total_moves > 0 else 0.5
            down_ratios_out[bar_count] = bar_down_moves / total_moves if total_moves > 0 else 0.5
            reversals_out[bar_count] = bar_reversals
            imbalances_out[bar_count] = (bar_buy_vol - bar_sell_vol) / bar_volume if bar_volume > 0.0 else 0.0
            max_vols_out[bar_count] = bar_max_vol
            max_ratios_out[bar_count] = bar_max_vol / bar_volume if bar_volume > 0.0 else 0.0
            interval_means_out[bar_count] = bar_interval_sum / bar_interval_count if bar_interval_count > 0 else 0.0
            net_move = bar_close - bar_open
            abs_net = net_move if net_move >= 0 else -net_move
            path_eff_out[bar_count] = abs_net / bar_path_length if bar_path_length > 0.0 else 0.0
            impact_den_out[bar_count] = abs_net / bar_dollar_vol if bar_dollar_vol > 0.0 else 0.0

        bar_count += 1

    # Truncate to actual bar count
    return (
        start_times[:bar_count], end_times[:bar_count],
        opens[:bar_count], highs[:bar_count], lows[:bar_count], closes[:bar_count],
        volumes[:bar_count], buy_vols[:bar_count], sell_vols[:bar_count],
        vwaps[:bar_count], tick_counts[:bar_count], dollar_vols[:bar_count],
        price_stds_out[:bar_count], volume_stds_out[:bar_count],
        up_ratios_out[:bar_count], down_ratios_out[:bar_count],
        reversals_out[:bar_count], imbalances_out[:bar_count],
        max_vols_out[:bar_count], max_ratios_out[:bar_count],
        interval_means_out[:bar_count],
        path_eff_out[:bar_count], impact_den_out[:bar_count],
    )


@jit(nopython=True, cache=True)
def _aggregate_dollar_bars_numba(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps: np.ndarray,
    is_buyer_maker: np.ndarray,
    threshold: float,
    include_advanced: bool,
    bar_mode: int,
) -> Tuple:
    """Single-pass Numba kernel for dollar/volume bar aggregation.

    Accumulates dollar volume (or raw volume) per bar and flushes when
    the running total exceeds ``threshold``.  All basic and advanced
    statistics are computed inline, identical to the time-bar kernel.

    Args:
        bar_mode: 0 = dollar_bar (accumulate price*qty), 1 = volume_bar (accumulate qty).

    Returns the same tuple layout as ``_aggregate_time_bars_numba``.
    """
    n = len(prices)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return (empty, empty, empty, empty, empty, empty,
                empty, empty, empty, empty, empty_i, empty,
                empty, empty, empty, empty, empty_i,
                empty, empty, empty, empty,
                empty, empty)

    # Estimate max bars: total_value / threshold + 1
    if bar_mode == 0:
        total_value = 0.0
        for i in range(n):
            total_value += prices[i] * quantities[i]
    else:
        total_value = 0.0
        for i in range(n):
            total_value += quantities[i]

    estimated_bars = int(total_value / threshold) + 2 if threshold > 0 else n
    max_bars = min(estimated_bars, n)

    # Pre-allocate output arrays
    start_times = np.empty(max_bars, dtype=np.float64)
    end_times = np.empty(max_bars, dtype=np.float64)
    opens = np.empty(max_bars, dtype=np.float64)
    highs = np.empty(max_bars, dtype=np.float64)
    lows = np.empty(max_bars, dtype=np.float64)
    closes = np.empty(max_bars, dtype=np.float64)
    volumes = np.empty(max_bars, dtype=np.float64)
    buy_vols = np.empty(max_bars, dtype=np.float64)
    sell_vols = np.empty(max_bars, dtype=np.float64)
    vwaps = np.empty(max_bars, dtype=np.float64)
    tick_counts = np.empty(max_bars, dtype=np.int64)
    dollar_vols = np.empty(max_bars, dtype=np.float64)
    # Advanced
    price_stds_out = np.empty(max_bars, dtype=np.float64)
    volume_stds_out = np.empty(max_bars, dtype=np.float64)
    up_ratios_out = np.empty(max_bars, dtype=np.float64)
    down_ratios_out = np.empty(max_bars, dtype=np.float64)
    reversals_out = np.empty(max_bars, dtype=np.int64)
    imbalances_out = np.empty(max_bars, dtype=np.float64)
    max_vols_out = np.empty(max_bars, dtype=np.float64)
    max_ratios_out = np.empty(max_bars, dtype=np.float64)
    interval_means_out = np.empty(max_bars, dtype=np.float64)
    path_eff_out = np.empty(max_bars, dtype=np.float64)
    impact_den_out = np.empty(max_bars, dtype=np.float64)

    bar_count = 0

    # Running accumulators
    bar_start_ts = timestamps[0]
    bar_end_ts = timestamps[0]
    bar_open = prices[0]
    bar_high = prices[0]
    bar_low = prices[0]
    bar_close = prices[0]
    bar_volume = 0.0
    bar_buy_vol = 0.0
    bar_sell_vol = 0.0
    bar_dollar_vol = 0.0
    bar_tick_count = 0
    bar_cum_value = 0.0  # cumulative value for threshold check
    # Advanced accumulators
    bar_price_sum = 0.0
    bar_price_sq_sum = 0.0
    bar_vol_sum = 0.0
    bar_vol_sq_sum = 0.0
    bar_up_moves = 0
    bar_down_moves = 0
    bar_prev_direction = 0
    bar_reversals = 0
    bar_max_vol = 0.0
    bar_path_length = 0.0
    bar_interval_sum = 0.0
    bar_interval_count = 0
    bar_prev_price = prices[0]

    for i in range(n):
        p = prices[i]
        q = quantities[i]

        # Accumulate tick into current bar
        bar_end_ts = timestamps[i]
        bar_close = p
        if p > bar_high:
            bar_high = p
        if p < bar_low:
            bar_low = p
        bar_volume += q
        bar_dollar_vol += p * q
        bar_tick_count += 1

        if bar_mode == 0:
            bar_cum_value += p * q
        else:
            bar_cum_value += q

        if is_buyer_maker[i]:
            bar_sell_vol += q
        else:
            bar_buy_vol += q

        if include_advanced:
            bar_price_sum += p
            bar_price_sq_sum += p * p
            bar_vol_sum += q
            bar_vol_sq_sum += q * q
            if q > bar_max_vol:
                bar_max_vol = q

            if bar_tick_count > 1:
                diff = p - bar_prev_price
                abs_diff = diff if diff >= 0 else -diff
                bar_path_length += abs_diff
                if diff > 0:
                    bar_up_moves += 1
                    direction = 1
                elif diff < 0:
                    bar_down_moves += 1
                    direction = -1
                else:
                    direction = bar_prev_direction
                if direction != 0 and bar_prev_direction != 0 and direction != bar_prev_direction:
                    bar_reversals += 1
                bar_prev_direction = direction

                interval_val = timestamps[i] - timestamps[i - 1]
                bar_interval_sum += interval_val
                bar_interval_count += 1

            bar_prev_price = p

        # Check if threshold reached → flush bar
        if bar_cum_value >= threshold:
            vwap_val = bar_dollar_vol / bar_volume if bar_volume > 0.0 else bar_close
            total_moves = bar_up_moves + bar_down_moves

            start_times[bar_count] = bar_start_ts
            end_times[bar_count] = bar_end_ts
            opens[bar_count] = bar_open
            highs[bar_count] = bar_high
            lows[bar_count] = bar_low
            closes[bar_count] = bar_close
            volumes[bar_count] = bar_volume
            buy_vols[bar_count] = bar_buy_vol
            sell_vols[bar_count] = bar_sell_vol
            vwaps[bar_count] = vwap_val
            tick_counts[bar_count] = bar_tick_count
            dollar_vols[bar_count] = bar_dollar_vol

            if include_advanced:
                pmean = bar_price_sum / bar_tick_count if bar_tick_count > 0 else 0.0
                pvar = bar_price_sq_sum / bar_tick_count - pmean * pmean if bar_tick_count > 0 else 0.0
                price_stds_out[bar_count] = np.sqrt(pvar) if pvar > 0.0 else 0.0
                vmean = bar_vol_sum / bar_tick_count if bar_tick_count > 0 else 0.0
                vvar = bar_vol_sq_sum / bar_tick_count - vmean * vmean if bar_tick_count > 0 else 0.0
                volume_stds_out[bar_count] = np.sqrt(vvar) if vvar > 0.0 else 0.0
                up_ratios_out[bar_count] = bar_up_moves / total_moves if total_moves > 0 else 0.5
                down_ratios_out[bar_count] = bar_down_moves / total_moves if total_moves > 0 else 0.5
                reversals_out[bar_count] = bar_reversals
                imbalances_out[bar_count] = (bar_buy_vol - bar_sell_vol) / bar_volume if bar_volume > 0.0 else 0.0
                max_vols_out[bar_count] = bar_max_vol
                max_ratios_out[bar_count] = bar_max_vol / bar_volume if bar_volume > 0.0 else 0.0
                interval_means_out[bar_count] = bar_interval_sum / bar_interval_count if bar_interval_count > 0 else 0.0
                net_move = bar_close - bar_open
                abs_net = net_move if net_move >= 0 else -net_move
                path_eff_out[bar_count] = abs_net / bar_path_length if bar_path_length > 0.0 else 0.0
                impact_den_out[bar_count] = abs_net / bar_dollar_vol if bar_dollar_vol > 0.0 else 0.0

            bar_count += 1

            # Reset accumulators for next bar
            if i + 1 < n:
                bar_start_ts = timestamps[i + 1] if i + 1 < n else timestamps[i]
                bar_open = prices[i + 1] if i + 1 < n else p
                bar_high = prices[i + 1] if i + 1 < n else p
                bar_low = prices[i + 1] if i + 1 < n else p
            bar_close = p
            bar_volume = 0.0
            bar_buy_vol = 0.0
            bar_sell_vol = 0.0
            bar_dollar_vol = 0.0
            bar_tick_count = 0
            bar_cum_value = 0.0
            bar_price_sum = 0.0
            bar_price_sq_sum = 0.0
            bar_vol_sum = 0.0
            bar_vol_sq_sum = 0.0
            bar_up_moves = 0
            bar_down_moves = 0
            bar_prev_direction = 0
            bar_reversals = 0
            bar_max_vol = 0.0
            bar_path_length = 0.0
            bar_interval_sum = 0.0
            bar_interval_count = 0
            bar_prev_price = p

    # Flush leftover partial bar if it has enough ticks (consistent with Pandas path)
    if bar_tick_count >= 2:
        vwap_val = bar_dollar_vol / bar_volume if bar_volume > 0.0 else bar_close
        total_moves = bar_up_moves + bar_down_moves

        start_times[bar_count] = bar_start_ts
        end_times[bar_count] = bar_end_ts
        opens[bar_count] = bar_open
        highs[bar_count] = bar_high
        lows[bar_count] = bar_low
        closes[bar_count] = bar_close
        volumes[bar_count] = bar_volume
        buy_vols[bar_count] = bar_buy_vol
        sell_vols[bar_count] = bar_sell_vol
        vwaps[bar_count] = vwap_val
        tick_counts[bar_count] = bar_tick_count
        dollar_vols[bar_count] = bar_dollar_vol

        if include_advanced:
            pmean = bar_price_sum / bar_tick_count if bar_tick_count > 0 else 0.0
            pvar = bar_price_sq_sum / bar_tick_count - pmean * pmean if bar_tick_count > 0 else 0.0
            price_stds_out[bar_count] = np.sqrt(pvar) if pvar > 0.0 else 0.0
            vmean = bar_vol_sum / bar_tick_count if bar_tick_count > 0 else 0.0
            vvar = bar_vol_sq_sum / bar_tick_count - vmean * vmean if bar_tick_count > 0 else 0.0
            volume_stds_out[bar_count] = np.sqrt(vvar) if vvar > 0.0 else 0.0
            up_ratios_out[bar_count] = bar_up_moves / total_moves if total_moves > 0 else 0.5
            down_ratios_out[bar_count] = bar_down_moves / total_moves if total_moves > 0 else 0.5
            reversals_out[bar_count] = bar_reversals
            imbalances_out[bar_count] = (bar_buy_vol - bar_sell_vol) / bar_volume if bar_volume > 0.0 else 0.0
            max_vols_out[bar_count] = bar_max_vol
            max_ratios_out[bar_count] = bar_max_vol / bar_volume if bar_volume > 0.0 else 0.0
            interval_means_out[bar_count] = bar_interval_sum / bar_interval_count if bar_interval_count > 0 else 0.0
            net_move = bar_close - bar_open
            abs_net = net_move if net_move >= 0 else -net_move
            path_eff_out[bar_count] = abs_net / bar_path_length if bar_path_length > 0.0 else 0.0
            impact_den_out[bar_count] = abs_net / bar_dollar_vol if bar_dollar_vol > 0.0 else 0.0

        bar_count += 1

    return (
        start_times[:bar_count], end_times[:bar_count],
        opens[:bar_count], highs[:bar_count], lows[:bar_count], closes[:bar_count],
        volumes[:bar_count], buy_vols[:bar_count], sell_vols[:bar_count],
        vwaps[:bar_count], tick_counts[:bar_count], dollar_vols[:bar_count],
        price_stds_out[:bar_count], volume_stds_out[:bar_count],
        up_ratios_out[:bar_count], down_ratios_out[:bar_count],
        reversals_out[:bar_count], imbalances_out[:bar_count],
        max_vols_out[:bar_count], max_ratios_out[:bar_count],
        interval_means_out[:bar_count],
        path_eff_out[:bar_count], impact_den_out[:bar_count],
    )


@jit(nopython=True, cache=True)
def _cumsum_threshold_indices(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Find indices where cumulative sum exceeds threshold (Numba-accelerated).
    
    Returns array of indices marking bar boundaries.
    """
    n = len(values)
    # Pre-allocate maximum possible size
    indices = np.empty(n, dtype=np.int64)
    count = 0
    cum_sum = 0.0
    
    for i in range(n):
        cum_sum += values[i]
        if cum_sum >= threshold:
            indices[count] = i
            count += 1
            cum_sum = 0.0
    
    return indices[:count]


@jit(nopython=True, cache=True)
def _compute_bar_stats(
    prices: np.ndarray,
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[float, float, float, float, float, float, float, float, int, float]:
    """
    Compute basic bar statistics (Numba-accelerated).
    
    Returns: (open, high, low, close, volume, buy_vol, sell_vol, vwap, tick_count, dollar_vol)
    """
    n = len(prices)
    
    open_p = prices[0]
    close_p = prices[n - 1]
    high_p = prices[0]
    low_p = prices[0]
    
    total_volume = 0.0
    buy_volume = 0.0
    sell_volume = 0.0
    dollar_volume = 0.0
    
    for i in range(n):
        p = prices[i]
        q = quantities[i]
        
        if p > high_p:
            high_p = p
        if p < low_p:
            low_p = p
        
        total_volume += q
        dollar_volume += p * q
        
        if is_buyer_maker[i]:
            sell_volume += q
        else:
            buy_volume += q
    
    vwap = dollar_volume / total_volume if total_volume > 0 else close_p
    
    return (open_p, high_p, low_p, close_p, total_volume, 
            buy_volume, sell_volume, vwap, n, dollar_volume)


@jit(nopython=True, cache=True)
def _compute_advanced_stats(
    prices: np.ndarray,
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[float, float, float, float, int, float, float, float, float, float, float]:
    """
    Compute advanced bar statistics (Numba-accelerated).

    Returns: (price_std, vol_std, up_ratio, down_ratio, reversals,
              imbalance, max_vol, max_ratio, interval_mean,
              path_efficiency, impact_density)
    """
    n = len(prices)

    # Price statistics
    price_sum = 0.0
    price_sq_sum = 0.0

    # Volume statistics
    vol_sum = 0.0
    vol_sq_sum = 0.0

    # Movement tracking
    up_moves = 0
    down_moves = 0
    prev_direction = 0
    reversals = 0

    # Path length: sum of |p_i - p_{i-1}| for Path Efficiency
    total_path_length = 0.0

    # Volume tracking
    max_vol = 0.0
    total_vol = 0.0
    buy_vol = 0.0
    sell_vol = 0.0

    # Dollar volume (for impact density)
    dollar_volume = 0.0

    # Time intervals
    interval_sum = 0.0
    interval_count = 0

    for i in range(n):
        p = prices[i]
        q = quantities[i]

        price_sum += p
        price_sq_sum += p * p
        vol_sum += q
        vol_sq_sum += q * q
        total_vol += q
        dollar_volume += p * q

        if q > max_vol:
            max_vol = q

        if is_buyer_maker[i]:
            sell_vol += q
        else:
            buy_vol += q

        if i > 0:
            diff = p - prices[i - 1]
            abs_diff = diff if diff >= 0 else -diff
            total_path_length += abs_diff

            if diff > 0:
                up_moves += 1
                direction = 1
            elif diff < 0:
                down_moves += 1
                direction = -1
            else:
                direction = prev_direction

            if direction != 0 and prev_direction != 0 and direction != prev_direction:
                reversals += 1
            prev_direction = direction

            # Time interval
            interval = timestamps[i] - timestamps[i - 1]
            interval_sum += interval
            interval_count += 1

    # Calculate standard deviations
    price_mean = price_sum / n
    price_var = price_sq_sum / n - price_mean * price_mean
    price_std = np.sqrt(price_var) if price_var > 0 else 0.0

    vol_mean = vol_sum / n
    vol_var = vol_sq_sum / n - vol_mean * vol_mean
    vol_std = np.sqrt(vol_var) if vol_var > 0 else 0.0

    # Ratios
    total_moves = up_moves + down_moves
    up_ratio = up_moves / total_moves if total_moves > 0 else 0.5
    down_ratio = down_moves / total_moves if total_moves > 0 else 0.5

    imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
    max_ratio = max_vol / total_vol if total_vol > 0 else 0.0

    interval_mean = interval_sum / interval_count if interval_count > 0 else 0.0

    # Path Efficiency: |close - open| / total_path_length
    net_move = prices[n - 1] - prices[0]
    abs_net_move = net_move if net_move >= 0 else -net_move
    path_efficiency = abs_net_move / total_path_length if total_path_length > 0 else 0.0

    # Intrabar Impact Density: |close - open| / dollar_volume
    impact_density = abs_net_move / dollar_volume if dollar_volume > 0 else 0.0

    return (price_std, vol_std, up_ratio, down_ratio, reversals,
            imbalance, max_vol, max_ratio, interval_mean,
            path_efficiency, impact_density)


# ============================================================================
# Fast Aggregator Class
# ============================================================================

@dataclass
class AggregationResult:
    """Result of aggregation operation."""
    symbol: str
    bar_type: BarType
    bars: pd.DataFrame
    source_file: Optional[str] = None
    processing_time: float = 0.0
    tick_count: int = 0
    bar_count: int = 0


class FastBarAggregator:
    """
    High-performance bar aggregator with Numba acceleration.
    
    Usage:
        aggregator = FastBarAggregator(use_numba=True, n_workers=4)
        
        # Single file
        bars = aggregator.aggregate_file(
            "data/ticks.parquet",
            BarType.DOLLAR_BAR,
            threshold=1000000
        )
        
        # Batch processing
        results = aggregator.aggregate_batch(
            file_paths,
            BarType.TIME_BAR,
            threshold="1h"
        )
    """
    
    def __init__(
        self,
        use_numba: bool = True,
        use_multiprocess: bool = True,
        n_workers: int = 4,
        include_advanced: bool = True
    ):
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_multiprocess = use_multiprocess
        self.n_workers = n_workers
        self.include_advanced = include_advanced
        
        if use_numba and not NUMBA_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning(
                "Numba not available, falling back to pure Python"
            )
    
    def aggregate(
        self,
        tick_data: pd.DataFrame,
        bar_type: Union[BarType, str],
        threshold: Union[int, float, str],
        symbol: str = "UNKNOWN"
    ) -> AggregationResult:
        """
        Aggregate tick data into bars.
        
        Args:
            tick_data: DataFrame with tick data
            bar_type: Type of bars to build
            threshold: Bar threshold
            symbol: Symbol name for the result
        
        Returns:
            AggregationResult with bars DataFrame
        """
        import time
        start_time = time.time()
        
        if isinstance(bar_type, str):
            bar_type = BarType(bar_type)
        
        if self.use_numba and bar_type == BarType.TIME_BAR:
            bars = self._aggregate_time_bars(tick_data, threshold)
        elif self.use_numba and bar_type in (BarType.VOLUME_BAR, BarType.DOLLAR_BAR):
            bars = self._aggregate_numba(tick_data, bar_type, threshold)
        else:
            bars = self._aggregate_pandas(tick_data, bar_type, threshold)
        
        processing_time = time.time() - start_time
        
        return AggregationResult(
            symbol=symbol,
            bar_type=bar_type,
            bars=bars,
            processing_time=processing_time,
            tick_count=len(tick_data),
            bar_count=len(bars),
        )
    
    def _aggregate_numba(
        self,
        data: pd.DataFrame,
        bar_type: BarType,
        threshold: Union[int, float]
    ) -> pd.DataFrame:
        """Aggregate dollar/volume bars using single-pass Numba kernel."""
        # Prepare contiguous numpy arrays
        prices = data["price"].values.astype(np.float64)
        quantities = data["quantity"].values.astype(np.float64)
        timestamps = data["timestamp"].values.astype(np.float64)

        if "is_buyer_maker" in data.columns:
            is_buyer = data["is_buyer_maker"].values.astype(bool)
        elif "isBuyerMaker" in data.columns:
            is_buyer = data["isBuyerMaker"].values.astype(bool)
        else:
            is_buyer = np.zeros(len(data), dtype=bool)

        # Ensure sorted by timestamp
        if len(timestamps) > 1 and not np.all(timestamps[:-1] <= timestamps[1:]):
            sort_order = np.argsort(timestamps)
            prices = prices[sort_order]
            quantities = quantities[sort_order]
            timestamps = timestamps[sort_order]
            is_buyer = is_buyer[sort_order]

        # bar_mode: 0 = dollar_bar, 1 = volume_bar
        bar_mode = 0 if bar_type == BarType.DOLLAR_BAR else 1

        result = _aggregate_dollar_bars_numba(
            prices, quantities, timestamps, is_buyer,
            float(threshold), self.include_advanced, bar_mode,
        )

        # Unpack and build DataFrame (same layout as time bar kernel)
        return self._unpack_bar_result(result)
    
    def _aggregate_time_bars(
        self,
        data: pd.DataFrame,
        threshold: Union[int, float, str],
    ) -> pd.DataFrame:
        """Aggregate time bars using the single-pass Numba kernel.

        Converts the threshold to milliseconds, prepares numpy arrays,
        calls ``_aggregate_time_bars_numba``, and packs the result arrays
        into a DataFrame.
        """
        interval_ms = _interval_to_ms(str(threshold))

        # Prepare contiguous numpy arrays (assumes data has been normalized by tick_normalizer)
        prices = data["price"].values.astype(np.float64)
        quantities = data["quantity"].values.astype(np.float64)
        timestamps = data["timestamp"].values.astype(np.float64)

        if "is_buyer_maker" in data.columns:
            is_buyer = data["is_buyer_maker"].values.astype(bool)
        elif "isBuyerMaker" in data.columns:
            is_buyer = data["isBuyerMaker"].values.astype(bool)
        else:
            is_buyer = np.zeros(len(data), dtype=bool)

        # Ensure data is sorted by timestamp (lightweight check first)
        if len(timestamps) > 1 and not np.all(timestamps[:-1] <= timestamps[1:]):
            sort_order = np.argsort(timestamps)
            prices = prices[sort_order]
            quantities = quantities[sort_order]
            timestamps = timestamps[sort_order]
            is_buyer = is_buyer[sort_order]

        result = _aggregate_time_bars_numba(
            prices, quantities, timestamps, is_buyer,
            interval_ms, self.include_advanced,
        )

        return self._unpack_bar_result(result)

    def _unpack_bar_result(self, result: Tuple) -> pd.DataFrame:
        """Convert Numba kernel output tuple into a pandas DataFrame.

        Shared by both time-bar and dollar/volume-bar kernels, which
        return the same 23-element tuple of numpy arrays.
        """
        (start_ts, end_ts, opens, highs, lows, closes,
         vols, buy_v, sell_v, vwap_arr, tick_cnts, dollar_v,
         p_stds, v_stds, up_r, down_r, revs, imbs,
         max_v, max_r, int_means, pe_arr, iid_arr) = result

        bar_count = len(start_ts)
        start_dts = np.empty(bar_count, dtype="datetime64[ms]")
        end_dts = np.empty(bar_count, dtype="datetime64[ms]")
        for idx in range(bar_count):
            start_dts[idx] = np.datetime64(int(start_ts[idx]), "ms")
            end_dts[idx] = np.datetime64(int(end_ts[idx]), "ms")

        bar_data: Dict[str, Any] = {
            "start_time": pd.to_datetime(start_dts, utc=True),
            "end_time": pd.to_datetime(end_dts, utc=True),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "buy_volume": buy_v,
            "sell_volume": sell_v,
            "vwap": vwap_arr,
            "tick_count": tick_cnts,
            "dollar_volume": dollar_v,
        }

        if self.include_advanced and bar_count > 0:
            bar_data.update({
                "price_std": p_stds,
                "volume_std": v_stds,
                "up_move_ratio": up_r,
                "down_move_ratio": down_r,
                "reversals": revs,
                "buy_sell_imbalance": imbs,
                "max_trade_volume": max_v,
                "max_trade_ratio": max_r,
                "tick_interval_mean": int_means,
                "path_efficiency": pe_arr,
                "impact_density": iid_arr,
            })

        return pd.DataFrame(bar_data)

    def _aggregate_pandas(
        self,
        data: pd.DataFrame,
        bar_type: BarType,
        threshold: Union[int, float, str]
    ) -> pd.DataFrame:
        """Aggregate using pandas-based bar builder."""
        config = BarConfig(
            bar_type=bar_type,
            threshold=threshold,
            include_advanced_features=self.include_advanced,
        )
        builder = get_bar_builder(config)
        return builder.build_bars(data)
    
    def _convert_timestamp(self, ts: float) -> datetime:
        """Convert timestamp to datetime."""
        ts_int = int(ts)
        length = len(str(ts_int))
        
        if length >= 16:
            ts_sec = ts / 1_000_000
        elif length == 13:
            ts_sec = ts / 1_000
        else:
            ts_sec = ts
        
        return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    
    def aggregate_file(
        self,
        file_path: Union[str, Path],
        bar_type: Union[BarType, str],
        threshold: Union[int, float, str],
        symbol: Optional[str] = None
    ) -> AggregationResult:
        """
        Aggregate a single file.
        
        Args:
            file_path: Path to tick data file (parquet or CSV)
            bar_type: Type of bars to build
            threshold: Bar threshold
            symbol: Symbol name (auto-detected from filename if None)
        
        Returns:
            AggregationResult with bars
        """
        file_path = Path(file_path)
        
        if symbol is None:
            symbol = file_path.stem
        
        # Load data
        if file_path.suffix == ".parquet":
            data = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        result = self.aggregate(data, bar_type, threshold, symbol)
        result.source_file = str(file_path)
        
        return result
    
    def aggregate_batch(
        self,
        file_paths: List[Union[str, Path]],
        bar_type: Union[BarType, str],
        threshold: Union[int, float, str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, AggregationResult]:
        """
        Aggregate multiple files in parallel.
        
        Args:
            file_paths: List of file paths
            bar_type: Type of bars to build
            threshold: Bar threshold
            progress_callback: Optional callback(completed, total)
        
        Returns:
            Dict mapping filename to AggregationResult
        """
        results = {}
        total = len(file_paths)
        completed = 0
        
        def process_file(path):
            return (path, self.aggregate_file(path, bar_type, threshold))
        
        if self.use_multiprocess and len(file_paths) > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(process_file, p): p for p in file_paths}
                
                for future in futures:
                    try:
                        path, result = future.result()
                        results[str(path)] = result
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Error processing {futures[future]}: {e}"
                        )
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
        else:
            for path in file_paths:
                try:
                    result = self.aggregate_file(path, bar_type, threshold)
                    results[str(path)] = result
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Error processing {path}: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        return results


class StreamingAggregator:
    """
    Memory-efficient streaming aggregator for very large datasets.
    
    Processes data in chunks to minimize memory usage.
    """
    
    def __init__(
        self,
        bar_type: BarType,
        threshold: Union[int, float, str],
        chunk_size: int = 1_000_000
    ):
        self.bar_type = bar_type
        self.threshold = threshold
        self.chunk_size = chunk_size
        self._buffer = []
        self._bars = []
    
    def process_chunk(self, chunk: pd.DataFrame) -> List[Dict]:
        """Process a chunk of tick data."""
        # Add to buffer
        self._buffer.append(chunk)
        
        # Concatenate buffer
        data = pd.concat(self._buffer, ignore_index=True)
        
        # Aggregate
        config = BarConfig(self.bar_type, self.threshold)
        builder = get_bar_builder(config)
        bars_df = builder.build_bars(data)
        
        if len(bars_df) > 0:
            # Store bars (except potentially incomplete last one)
            complete_bars = bars_df.iloc[:-1].to_dict("records")
            self._bars.extend(complete_bars)
            
            # Find remaining data after last COMPLETE bar (not the discarded one)
            if len(bars_df) > 1:
                last_complete_end = bars_df["end_time"].iloc[-2]
                if "timestamp" in data.columns:
                    remaining = data[data["timestamp"] > last_complete_end.timestamp() * 1000]
                else:
                    remaining = pd.DataFrame()
            else:
                # Only one bar produced (which we discard as incomplete), keep all data
                remaining = data
            
            # Update buffer
            self._buffer = [remaining] if len(remaining) > 0 else []
            
            return complete_bars
        
        return []
    
    def finalize(self) -> pd.DataFrame:
        """Finalize aggregation and return all bars."""
        # Process any remaining buffer
        if self._buffer:
            data = pd.concat(self._buffer, ignore_index=True)
            config = BarConfig(self.bar_type, self.threshold)
            builder = get_bar_builder(config)
            bars_df = builder.build_bars(data)
            self._bars.extend(bars_df.to_dict("records"))
        
        return pd.DataFrame(self._bars)
    
    def reset(self) -> None:
        """Reset the aggregator state."""
        self._buffer = []
        self._bars = []
