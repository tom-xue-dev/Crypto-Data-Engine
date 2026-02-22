"""
Tick Microstructure Factor Calculation Module.

Extracts features directly from raw tick (aggTrades) data that cannot be
derived from bar-level aggregations. These capture intra-day dynamics that
are lost during bar construction.

Factors:
1. VPIN — Volume-Synchronized Probability of Informed Trading
2. Order Flow Toxicity — Consecutive same-direction trade run statistics
3. Kyle Lambda — Precise tick-level price impact regression
4. Burstiness — Trade arrival dynamics (Hawkes-like clustering)
5. Jump Ratio — Bipower Variation vs Realized Variance
6. Whale Activity — Large-trade directional imbalance + permanent impact

Input: Raw aggTrades DataFrame with columns:
    - price: float64
    - quantity: float64
    - transact_time: int64 (ms epoch)
    - is_buyer_maker: bool (True = seller-initiated)

References:
    - Easley, López de Prado & O'Hara (2012) — VPIN
    - Kyle (1985), Hasbrouck (2009) — Kyle Lambda
    - Barndorff-Nielsen & Shephard (2006) — Jump Detection
    - Cont, Kukanov & Stoikov (2014) — Whale Activity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TickMicrostructureConfig:
    """Configuration for tick microstructure factor extraction."""

    # VPIN
    vpin_n_buckets: int = 50

    # Kyle Lambda
    kyle_freq_ms: int = 60_000   # 1-minute buckets for regression
    kyle_min_buckets: int = 10

    # Jump Detection
    jump_freq_ms: int = 300_000  # 5-minute resampling
    jump_min_intervals: int = 5

    # Whale Activity
    whale_percentile: int = 99
    whale_impact_delay_ms: int = 30_000  # 30s permanent impact window

    # Minimum ticks per day to compute features
    min_ticks_per_day: int = 500


# ══════════════════════════════════════════════════════════════════
# INDIVIDUAL FACTOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════


def compute_vpin(
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    n_buckets: int = 50,
) -> float:
    """
    Volume-Synchronized Probability of Informed Trading.

    Splits trades into equal-volume buckets, computes
    |buy_vol - sell_vol| / total_vol per bucket, returns mean.

    Returns:
        Scalar in [0, 1]. 1 = fully one-sided, 0 = balanced.
    """
    if len(quantities) < n_buckets * 2:
        return np.nan

    buy_vol = quantities.copy().astype(float)
    sell_vol = quantities.copy().astype(float)
    buy_vol[is_buyer_maker] = 0.0
    sell_vol[~is_buyer_maker] = 0.0

    total_vol = quantities.sum()
    if total_vol <= 0:
        return np.nan
    bucket_size = total_vol / n_buckets

    cum_vol = np.cumsum(quantities.astype(float))
    cum_buy = np.cumsum(buy_vol)
    cum_sell = np.cumsum(sell_vol)

    boundaries = np.arange(1, n_buckets + 1) * bucket_size
    bucket_idx = np.searchsorted(cum_vol, boundaries, side="right")
    bucket_idx = np.minimum(bucket_idx, len(quantities) - 1)

    vpin_vals = []
    prev_cum_buy = 0.0
    prev_cum_sell = 0.0
    prev_idx = 0
    for bi in bucket_idx:
        if bi <= prev_idx and len(vpin_vals) > 0:
            continue
        b_buy = cum_buy[bi] - prev_cum_buy
        b_sell = cum_sell[bi] - prev_cum_sell
        b_total = b_buy + b_sell
        if b_total > 0:
            vpin_vals.append(abs(b_buy - b_sell) / b_total)
        prev_cum_buy = cum_buy[bi]
        prev_cum_sell = cum_sell[bi]
        prev_idx = bi

    if not vpin_vals:
        return np.nan
    return float(np.mean(vpin_vals))


def compute_toxicity(
    is_buyer_maker: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Order Flow Toxicity — run statistics of consecutive same-direction trades.

    Returns:
        (run_mean, run_max, toxicity_ratio)
        - run_mean: average consecutive same-direction run length
        - run_max: longest run
        - toxicity_ratio: P(same direction | previous direction)
    """
    n = len(is_buyer_maker)
    if n < 10:
        return np.nan, np.nan, np.nan

    direction = (~is_buyer_maker).astype(np.int8)
    changes = np.diff(direction)
    change_points = np.where(changes != 0)[0] + 1
    run_starts = np.concatenate([[0], change_points])
    run_ends = np.concatenate([change_points, [n]])
    run_lengths = run_ends - run_starts

    if len(run_lengths) == 0:
        return np.nan, np.nan, np.nan

    run_mean = float(np.mean(run_lengths))
    run_max = int(np.max(run_lengths))

    n_continuations = np.sum(changes == 0)
    toxicity_ratio = float(n_continuations / len(changes)) if len(changes) > 0 else 0.0

    return run_mean, run_max, toxicity_ratio


def compute_kyle_lambda(
    prices: np.ndarray,
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    timestamps: np.ndarray,
    freq_ms: int = 60_000,
    min_buckets: int = 10,
) -> float:
    """
    Kyle Lambda — price impact per unit signed volume.

    Buckets trades by time, regresses ΔP on signed √volume.
    λ = Cov(ΔP, sign(Q)√|Q|) / Var(sign(Q)√|Q|)

    Returns:
        Scalar λ (higher = less liquid / more price impact).
    """
    n = len(prices)
    if n < 100:
        return np.nan

    signed_qty = quantities.astype(float).copy()
    signed_qty[is_buyer_maker] *= -1

    t_start, t_end = timestamps[0], timestamps[-1]
    if t_end - t_start < freq_ms * min_buckets:
        return np.nan

    bucket_edges = np.arange(t_start, t_end + freq_ms, freq_ms)
    bucket_assign = np.searchsorted(bucket_edges, timestamps, side="right") - 1

    n_buckets = len(bucket_edges) - 1
    if n_buckets < min_buckets:
        return np.nan

    # Vectorized: find bucket boundaries via change-points (timestamps sorted)
    dps = np.full(n_buckets, np.nan)
    svs = np.full(n_buckets, np.nan)

    changes = np.where(np.diff(bucket_assign) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [n]])
    b_ids = bucket_assign[starts]

    for k in range(len(starts)):
        s, e = starts[k], ends[k]
        if e - s < 2:
            continue
        b = b_ids[k]
        if b < 0 or b >= n_buckets:
            continue
        bp = prices[s:e]
        dps[b] = (bp[-1] - bp[0]) / bp[0] * 1e4
        svs[b] = np.sum(signed_qty[s:e])

    valid = np.isfinite(dps) & np.isfinite(svs)
    if valid.sum() < min_buckets:
        return np.nan

    dp_v = dps[valid]
    sv_v = svs[valid]
    x = np.sign(sv_v) * np.sqrt(np.abs(sv_v))
    var_x = np.var(x, ddof=1)
    if var_x < 1e-20:
        return np.nan

    cov_xy = np.mean((dp_v - np.mean(dp_v)) * (x - np.mean(x)))
    return float(cov_xy / var_x)


def compute_burstiness(timestamps: np.ndarray) -> float:
    """
    Burstiness of trade arrivals.

    B = (σ_ITI - μ_ITI) / (σ_ITI + μ_ITI)

    Returns:
        Scalar in [-1, 1]. -1 = periodic, 0 = Poisson, +1 = very bursty.
    """
    if len(timestamps) < 20:
        return np.nan

    iti = np.diff(timestamps).astype(float)
    iti = iti[iti > 0]
    if len(iti) < 10:
        return np.nan

    mean_iti = np.mean(iti)
    std_iti = np.std(iti, ddof=1)
    denom = std_iti + mean_iti
    if denom < 1e-10:
        return -1.0

    return float((std_iti - mean_iti) / denom)


def compute_jump_ratio(
    prices: np.ndarray,
    timestamps: np.ndarray,
    freq_ms: int = 300_000,
    min_intervals: int = 5,
) -> float:
    """
    Jump Detection — Bipower Variation vs Realized Variance.

    Resamples to freq_ms intervals, computes log returns, then:
      RV = Σ r_i²
      BV = (π/2) × Σ |r_i| × |r_{i-1}|
      jump_ratio = max(0, 1 - BV/RV)

    Returns:
        Scalar in [0, 1]. Higher = more jump activity.
    """
    n = len(prices)
    if n < 50:
        return np.nan

    t_start, t_end = timestamps[0], timestamps[-1]
    if t_end - t_start < freq_ms * min_intervals:
        return np.nan

    bucket_edges = np.arange(t_start, t_end + freq_ms, freq_ms)
    n_buckets = len(bucket_edges) - 1
    bucket_assign = np.searchsorted(bucket_edges, timestamps, side="right") - 1

    # Vectorized: get last price per bucket using change-points
    closes = np.full(n_buckets, np.nan)
    changes = np.where(np.diff(bucket_assign) != 0)[0]
    # Last index before each change = last in that bucket
    last_indices = np.concatenate([changes, [n - 1]])
    b_ids = bucket_assign[last_indices]
    for k in range(len(last_indices)):
        b = b_ids[k]
        if 0 <= b < n_buckets:
            closes[b] = prices[last_indices[k]]

    # Forward-fill
    for i in range(1, len(closes)):
        if not np.isfinite(closes[i]) and np.isfinite(closes[i - 1]):
            closes[i] = closes[i - 1]

    valid_closes = closes[np.isfinite(closes)]
    if len(valid_closes) < min_intervals:
        return np.nan

    log_ret = np.diff(np.log(valid_closes))
    log_ret = log_ret[np.isfinite(log_ret)]
    if len(log_ret) < 3:
        return np.nan

    rv = np.sum(log_ret ** 2)
    if rv < 1e-20:
        return 0.0

    abs_ret = np.abs(log_ret)
    bv = (np.pi / 2) * np.sum(abs_ret[1:] * abs_ret[:-1])
    return float(max(0.0, 1.0 - bv / rv))


def compute_whale_metrics(
    prices: np.ndarray,
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    timestamps: np.ndarray,
    percentile: int = 99,
    impact_delay_ms: int = 30_000,
) -> Tuple[float, float]:
    """
    Whale Activity — large-trade directional imbalance + permanent impact.

    Returns:
        (imbalance, permanent_impact)
        - imbalance: (whale_buy - whale_sell) / whale_total in [-1, 1]
        - permanent_impact: mean signed price change after whale trades
    """
    n = len(prices)
    if n < 100:
        return np.nan, np.nan

    threshold = np.percentile(quantities, percentile)
    whale_mask = quantities >= threshold
    if whale_mask.sum() < 3:
        return np.nan, np.nan

    whale_qty = quantities[whale_mask]
    whale_ibm = is_buyer_maker[whale_mask]

    whale_buy_vol = whale_qty[~whale_ibm].sum()
    whale_sell_vol = whale_qty[whale_ibm].sum()
    total = whale_buy_vol + whale_sell_vol
    imbalance = float((whale_buy_vol - whale_sell_vol) / total) if total > 0 else 0.0

    whale_indices = np.where(whale_mask)[0]
    # Vectorized impact: find price at t+delay for all whales at once
    t_afters = timestamps[whale_indices] + impact_delay_ms
    after_idxs = np.searchsorted(timestamps, t_afters, side="right") - 1

    valid_mask = (after_idxs > whale_indices) & (after_idxs < n)
    if valid_mask.sum() < 3:
        return imbalance, np.nan

    wi_valid = whale_indices[valid_mask]
    ai_valid = after_idxs[valid_mask]
    dp = (prices[ai_valid] - prices[wi_valid]) / prices[wi_valid]
    # Flip sign for seller-initiated whales
    seller_mask = is_buyer_maker[wi_valid]
    dp[seller_mask] = -dp[seller_mask]

    perm_impact = float(np.mean(dp))
    return imbalance, perm_impact


# ══════════════════════════════════════════════════════════════════
# DAILY EXTRACTION WRAPPER
# ══════════════════════════════════════════════════════════════════


def extract_daily_features(
    df: pd.DataFrame,
    config: Optional[TickMicrostructureConfig] = None,
) -> Dict[str, float]:
    """
    Extract all 6 tick microstructure features from one day's tick data.

    Args:
        df: DataFrame with columns [price, quantity, transact_time, is_buyer_maker]
        config: Optional configuration overrides

    Returns:
        Dict with keys: vpin, toxicity_run_mean, toxicity_run_max,
        toxicity_ratio, kyle_lambda, burstiness, jump_ratio,
        whale_imbalance, whale_perm_impact
    """
    cfg = config or TickMicrostructureConfig()

    prices = df["price"].values.astype(float)
    quantities = df["quantity"].values.astype(float)
    timestamps = df["transact_time"].values.astype(np.int64)
    is_buyer_maker = df["is_buyer_maker"].values.astype(bool)

    vpin = compute_vpin(quantities, is_buyer_maker, cfg.vpin_n_buckets)
    run_mean, run_max, tox_ratio = compute_toxicity(is_buyer_maker)
    kyle_lam = compute_kyle_lambda(
        prices, quantities, is_buyer_maker, timestamps,
        cfg.kyle_freq_ms, cfg.kyle_min_buckets,
    )
    burst = compute_burstiness(timestamps)
    jump = compute_jump_ratio(
        prices, timestamps, cfg.jump_freq_ms, cfg.jump_min_intervals,
    )
    whale_imb, whale_pi = compute_whale_metrics(
        prices, quantities, is_buyer_maker, timestamps,
        cfg.whale_percentile, cfg.whale_impact_delay_ms,
    )

    return dict(
        vpin=vpin,
        toxicity_run_mean=run_mean,
        toxicity_run_max=run_max,
        toxicity_ratio=tox_ratio,
        kyle_lambda=kyle_lam,
        burstiness=burst,
        jump_ratio=jump,
        whale_imbalance=whale_imb,
        whale_perm_impact=whale_pi,
    )


# ══════════════════════════════════════════════════════════════════
# BATCH / MULTIPROCESSING SUPPORT
# ══════════════════════════════════════════════════════════════════


def process_tick_file(
    file_path: str,
    min_ticks_per_day: int = 500,
    config: Optional[TickMicrostructureConfig] = None,
) -> List[Dict]:
    """
    Process one monthly parquet file → list of daily feature dicts.

    Designed for use with multiprocessing (ProcessPoolExecutor).

    Args:
        file_path: Path to aggTrades parquet file
        min_ticks_per_day: Minimum ticks required to compute features
        config: Optional configuration overrides

    Returns:
        List of dicts, each with 'date' key + feature values.
    """
    try:
        df = pd.read_parquet(
            file_path,
            columns=["price", "quantity", "transact_time", "is_buyer_maker"],
        )
    except Exception:
        return []

    if len(df) < min_ticks_per_day:
        return []

    df["date"] = pd.to_datetime(df["transact_time"], unit="ms").dt.date
    results = []

    for date, day_df in df.groupby("date"):
        if len(day_df) < min_ticks_per_day:
            continue
        feats = extract_daily_features(day_df, config)
        feats["date"] = date
        results.append(feats)

    return results
