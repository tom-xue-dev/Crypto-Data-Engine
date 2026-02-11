"""
Dollar Bar Cross-Sectional Backtest with Dynamic Asset Pool.

Memory-efficient pipeline using segmented backtesting:
  - Splits the full time range into segments (default 6 months)
  - Each segment: load data -> calculate features -> run backtest -> free memory
  - NAV and trades are chained across segments
  - Asset pool is dynamically selected from profiles (daily dollar_volume)

Strategies:
  - momentum_20: Long top momentum, short bottom
  - mean_reversion: Long recent losers, short recent winners
  - path_efficiency: Long strong trends, short weak trends
  - order_flow: Long buy pressure, short sell pressure
  - volatility_regime: Long high VPIN (informed trading), short low
  - multi_factor: Composite of momentum + path_efficiency + order_flow

Usage:
    python scripts/run_dollarbar_backtest.py
    python scripts/run_dollarbar_backtest.py --bar-dir E:/data/dollar_bar/bars
    python scripts/run_dollarbar_backtest.py --pool-top-n 50 --long-n 5 --short-n 5
    python scripts/run_dollarbar_backtest.py --strategy momentum_20
    python scripts/run_dollarbar_backtest.py --verify-only
"""
import argparse
import gc
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src is on the import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
)
logger = logging.getLogger("dollarbar_backtest")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "start_time", "end_time", "open", "high", "low", "close",
    "volume", "buy_volume", "sell_volume", "vwap", "dollar_volume",
]

# Factors using mean-reversion logic: long losers, short winners
REVERSAL_FACTORS = frozenset({"return_5", "mean_reversion"})

# Available strategies and their configurations
STRATEGY_CONFIGS = {
    "momentum_20": {
        "factor": "momentum_20",
        "type": "momentum",
        "description": "Long top momentum, short bottom",
    },
    "mean_reversion": {
        "factor": "return_5",
        "type": "mean_reversion",
        "description": "Long recent losers, short recent winners",
    },
    "path_efficiency": {
        "factor": "signed_pe_20",
        "type": "momentum",
        "description": "Long strong trends, short weak trends",
    },
    "order_flow": {
        "factor": "OFI_approx_20",
        "type": "momentum",
        "description": "Long buy pressure, short sell pressure",
    },
    "volatility_regime": {
        "factor": "VPIN_50",
        "type": "momentum",
        "description": "Long high VPIN (informed trading), short low",
    },
    "multi_factor": {
        "factor": "multi_factor_score",
        "type": "momentum",
        "description": "Composite: momentum + path_efficiency + order_flow",
    },
}


# ---------------------------------------------------------------------------
# Helpers: file-level month scanning (no data loading)
# ---------------------------------------------------------------------------

_YEAR_MONTH_PATTERN = re.compile(r"(\d{4})-(\d{2})\.parquet$")


def _extract_year_month(filename: str) -> Optional[Tuple[int, int]]:
    """Extract (year, month) from a bar parquet filename."""
    match = _YEAR_MONTH_PATTERN.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def scan_available_months(bar_dir: str, min_months: int = 6) -> Dict[str, List[Path]]:
    """Scan bar directory and return symbol -> sorted list of parquet files.

    Only includes symbols with at least *min_months* files.
    Does NOT load any data into memory.
    """
    bar_path = Path(bar_dir)
    if not bar_path.exists():
        raise FileNotFoundError(f"Bar directory not found: {bar_dir}")

    symbol_dirs = sorted(d for d in bar_path.iterdir() if d.is_dir())
    logger.info(f"Found {len(symbol_dirs)} symbol directories in {bar_dir}")

    symbol_files: Dict[str, List[Path]] = {}
    skipped = 0

    for symbol_dir in symbol_dirs:
        files = sorted(symbol_dir.glob("*.parquet"))
        if len(files) < min_months:
            skipped += 1
            continue
        symbol_files[symbol_dir.name] = files

    logger.info(
        f"Eligible symbols: {len(symbol_files)} (skipped {skipped} with < {min_months} months)"
    )
    return symbol_files


def get_global_month_range(
    symbol_files: Dict[str, List[Path]],
) -> List[Tuple[int, int]]:
    """Return a sorted list of unique (year, month) tuples across all symbols."""
    months_set: set = set()
    for files in symbol_files.values():
        for filepath in files:
            year_month = _extract_year_month(filepath.name)
            if year_month:
                months_set.add(year_month)
    return sorted(months_set)


def build_segments(
    all_months: List[Tuple[int, int]],
    segment_months: int = 6,
    warmup_months: int = 2,
) -> List[Dict]:
    """Split the month list into overlapping segments for chunked backtesting.

    Each segment dict has:
        load_months  - months to load (includes warmup for feature calculation)
        eval_months  - months to actually evaluate in the backtest
    """
    if not all_months:
        return []

    segments: List[Dict] = []
    total = len(all_months)
    index = 0

    while index < total:
        eval_end = min(index + segment_months, total)
        eval_months = all_months[index:eval_end]

        # Warmup: go back *warmup_months* before eval_months[0]
        warmup_start = max(0, index - warmup_months)
        load_months = all_months[warmup_start:eval_end]

        segments.append({
            "load_months": load_months,
            "eval_months": eval_months,
        })
        index = eval_end

    return segments


# ---------------------------------------------------------------------------
# Asset Pool Management (using profiles)
# ---------------------------------------------------------------------------

def load_asset_pool_from_profiles(
    profiles_dir: str,
    target_month: Tuple[int, int],
    top_n: int = 100,
) -> List[str]:
    """
    Load asset pool from profiles directory based on previous month's dollar volume.

    Args:
        profiles_dir: Directory containing {SYMBOL}_daily_profile.parquet files
        target_month: (year, month) tuple for which we want the pool
        top_n: Number of top symbols to select

    Returns:
        List of top_n symbols by average daily dollar volume
    """
    profiles_path = Path(profiles_dir)
    if not profiles_path.exists():
        logger.warning(f"Profiles directory not found: {profiles_dir}")
        return []

    # Calculate previous month for lookback
    year, month = target_month
    if month == 1:
        prev_year, prev_month = year - 1, 12
    else:
        prev_year, prev_month = year, month - 1

    prev_start = pd.Timestamp(year=prev_year, month=prev_month, day=1, tz="UTC")
    prev_end = pd.Timestamp(year=year, month=month, day=1, tz="UTC") - pd.Timedelta(days=1)

    symbol_volumes: Dict[str, float] = {}
    profile_files = list(profiles_path.glob("*_daily_profile.parquet"))

    for profile_file in profile_files:
        symbol = profile_file.stem.replace("_daily_profile", "")
        try:
            df = pd.read_parquet(profile_file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True)
                mask = (df["date"] >= prev_start) & (df["date"] <= prev_end)
                period_data = df[mask]
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                mask = (df["timestamp"] >= prev_start) & (df["timestamp"] <= prev_end)
                period_data = df[mask]
            else:
                # Try with index
                if isinstance(df.index, pd.DatetimeIndex):
                    mask = (df.index >= prev_start) & (df.index <= prev_end)
                    period_data = df[mask]
                else:
                    period_data = df

            if len(period_data) > 0:
                # Find dollar volume column
                vol_col = None
                for col in ["dollar_volume", "daily_dollar_volume", "dollar_vol", "volume_usd"]:
                    if col in period_data.columns:
                        vol_col = col
                        break
                if vol_col:
                    avg_vol = period_data[vol_col].mean()
                    if pd.notna(avg_vol) and avg_vol > 0:
                        symbol_volumes[symbol] = avg_vol
        except Exception as e:
            logger.debug(f"  Skip profile {profile_file.name}: {e}")

    # Sort by volume and return top_n
    sorted_symbols = sorted(symbol_volumes.items(), key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_symbols[:top_n]]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_bar_data_for_months(
    symbol_files: Dict[str, List[Path]],
    months_to_load: List[Tuple[int, int]],
    asset_pool: Optional[List[str]] = None,
    max_symbols: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Load bar data for specific months only, keeping memory low.

    Args:
        symbol_files: Pre-scanned symbol -> file list mapping.
        months_to_load: Which (year, month) files to load.
        asset_pool: If provided, only load these symbols.
        max_symbols: Cap on number of symbols (0 = all).

    Returns:
        Dict mapping symbol -> merged DataFrame (only the requested months).
    """
    month_set = set(months_to_load)
    loaded: Dict[str, pd.DataFrame] = {}

    # Filter to asset pool if provided
    symbols_to_load = list(symbol_files.keys())
    if asset_pool:
        symbols_to_load = [s for s in symbols_to_load if s in asset_pool]

    for symbol in symbols_to_load:
        if max_symbols > 0 and len(loaded) >= max_symbols:
            break

        files = symbol_files.get(symbol, [])
        frames = []
        for filepath in files:
            year_month = _extract_year_month(filepath.name)
            if year_month and year_month in month_set:
                try:
                    dataframe = pd.read_parquet(filepath)
                    if len(dataframe) > 0:
                        frames.append(dataframe)
                except Exception as error:
                    logger.debug(f"  Skip {filepath.name}: {error}")

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)
        if "start_time" in merged.columns:
            merged["start_time"] = pd.to_datetime(merged["start_time"], utc=True)
            merged = merged.sort_values("start_time").reset_index(drop=True)

        # Verify required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in merged.columns]
        if missing:
            logger.debug(f"  {symbol}: missing columns {missing}")
            continue

        loaded[symbol] = merged

    return loaded


# ---------------------------------------------------------------------------
# Feature Calculation
# ---------------------------------------------------------------------------

def calculate_features(
    bar_data: Dict[str, pd.DataFrame],
    feature_windows: List[int],
    eval_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Calculate features for all symbols, optionally trim to eval window.

    Args:
        bar_data: Symbol -> bar DataFrame.
        feature_windows: Rolling window sizes.
        eval_start: If set, drop rows before this timestamp after feature calc
                    (removes warmup-only rows to save memory).

    Returns:
        MultiIndex DataFrame (time, asset).
    """
    from crypto_data_engine.services.feature.unified_features import (
        UnifiedFeatureCalculator,
        UnifiedFeatureConfig,
    )

    config = UnifiedFeatureConfig(
        windows=feature_windows,
        include_returns=True,
        include_volatility=True,
        include_momentum=True,
        include_volume=True,
        include_microstructure=True,
        include_alphas=False,
        include_technical=False,
        include_cross_sectional=False,
    )
    calculator = UnifiedFeatureCalculator(config)

    frames: List[pd.DataFrame] = []
    failed_count = 0

    for asset, bars in bar_data.items():
        try:
            if len(bars) < 100:
                continue

            # Calculate unified features
            featured = calculator.calculate(bars, asset=asset)

            # Add dollar-bar specific factors
            featured = _add_dollar_bar_factors(featured)

            featured["asset"] = asset

            # Find time column for this chunk
            time_col = next(
                (c for c in ["start_time", "timestamp", "time"] if c in featured.columns),
                None,
            )
            if time_col and eval_start is not None:
                featured[time_col] = pd.to_datetime(featured[time_col], utc=True)
                featured = featured[featured[time_col] >= eval_start]

            if len(featured) > 0:
                frames.append(featured)
        except Exception as error:
            logger.debug(f"  {asset}: feature calculation failed - {error}")
            failed_count += 1

    if not frames:
        raise RuntimeError("No assets produced valid features in this segment")

    combined = pd.concat(frames, ignore_index=True)

    # Standardize time column
    time_col = next(
        (c for c in ["start_time", "timestamp", "time"] if c in combined.columns),
        None,
    )
    if time_col is None:
        raise RuntimeError(f"No time column found. Columns: {list(combined.columns)}")

    # For dollar bars, round to nearest hour for cross-sectional alignment
    combined[time_col] = pd.to_datetime(combined[time_col], utc=True).dt.floor("1h")
    combined = combined.set_index([time_col, "asset"]).sort_index()

    asset_count = combined.index.get_level_values("asset").nunique()
    logger.info(
        f"Features: {combined.shape[0]:,} rows x {combined.shape[1]} cols, "
        f"{asset_count} assets (failed: {failed_count})"
    )
    return combined


def _add_dollar_bar_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add dollar-bar specific factors (VPIN, Kyle Lambda, MR, OFI, etc.)."""
    out = df.copy()

    # VPIN(50): mean(|buy_volume - sell_volume| / volume) over 50 bars
    if "buy_volume" in out.columns and "sell_volume" in out.columns and "volume" in out.columns:
        vpin_bar = (out["buy_volume"] - out["sell_volume"]).abs() / out["volume"].replace(0, np.nan)
        out["VPIN_50"] = vpin_bar.rolling(50, min_periods=20).mean()

    # Kyle Lambda(100): sum(|dclose|) / sum(dollar_volume)
    if "close" in out.columns and "dollar_volume" in out.columns:
        abs_ret = out["close"].diff().abs()
        out["Kyle_Lambda_100"] = (
            abs_ret.rolling(100, min_periods=40).sum()
            / out["dollar_volume"].rolling(100, min_periods=40).sum().replace(0, np.nan)
        )

    # MR(20, 200): (VWAP_fast - VWAP_slow) / realized_vol_slow
    if "vwap" in out.columns and "close" in out.columns:
        vwap_fast = out["vwap"].rolling(20, min_periods=10).mean()
        vwap_slow = out["vwap"].rolling(200, min_periods=100).mean()
        vol_slow = out["close"].pct_change().rolling(200, min_periods=100).std()
        out["MR_20_200"] = (vwap_fast - vwap_slow) / vol_slow.replace(0, np.nan)

    # Trade Intensity: tick_count / time_span (per bar then rolling), then z-score
    if "tick_count" in out.columns and "end_time" in out.columns and "start_time" in out.columns:
        time_span_sec = (
            pd.to_datetime(out["end_time"]) - pd.to_datetime(out["start_time"])
        ).dt.total_seconds().replace(0, np.nan)
        lambda_20 = out["tick_count"].rolling(20, min_periods=10).sum() / time_span_sec.rolling(20, min_periods=10).sum()
        out["lambda_20"] = lambda_20
        roll500_mean = lambda_20.rolling(500, min_periods=100).mean()
        roll500_std = lambda_20.rolling(500, min_periods=100).std()
        out["lambda_20_zscore"] = (lambda_20 - roll500_mean) / roll500_std.replace(0, np.nan)

    # Approximate OFI: (buy_volume - sell_volume) / (buy_volume + sell_volume), then rolling
    if "buy_volume" in out.columns and "sell_volume" in out.columns:
        ofi_bar = (out["buy_volume"] - out["sell_volume"]) / (
            (out["buy_volume"] + out["sell_volume"]).replace(0, np.nan)
        )
        for w in [20, 100]:
            col = f"OFI_approx_{w}"
            out[col] = ofi_bar.rolling(w, min_periods=w // 2).mean()

    # Multi-factor composite score (z-scored then averaged)
    _add_multi_factor_score(out)

    return out


def _add_multi_factor_score(df: pd.DataFrame) -> None:
    """Add composite multi-factor score by z-scoring and averaging factors."""
    factor_cols = []

    # Momentum component
    if "momentum_20" in df.columns:
        factor_cols.append("momentum_20")

    # Path efficiency component
    if "signed_pe_20" in df.columns:
        factor_cols.append("signed_pe_20")

    # Order flow component
    if "OFI_approx_20" in df.columns:
        factor_cols.append("OFI_approx_20")

    if len(factor_cols) >= 2:
        # Z-score each factor over rolling window
        z_scores = []
        for col in factor_cols:
            roll_mean = df[col].rolling(100, min_periods=50).mean()
            roll_std = df[col].rolling(100, min_periods=50).std()
            z = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
            z_scores.append(z)

        # Average z-scores
        df["multi_factor_score"] = pd.concat(z_scores, axis=1).mean(axis=1)


# ---------------------------------------------------------------------------
# Backtest Execution
# ---------------------------------------------------------------------------

def _create_engine_for_factor(
    factor: str,
    capital: float,
    timestamps_min,
    timestamps_max,
    rebalance_frequency: str,
    top_n_long: int,
    top_n_short: int,
    pool_top_n: int,
    pool_reselect: str,
    pool_lookback: str,
    strategy_type: str = "momentum",
):
    """Create a fresh backtest engine + strategy for one factor."""
    from crypto_data_engine.services.back_test import (
        BacktestConfig,
        BacktestMode,
        CostConfigModel,
        RiskConfigModel,
        TradingLogger,
        create_backtest_engine,
        create_strategy,
    )
    from crypto_data_engine.services.back_test.config import (
        AssetPoolConfig as BtAssetPoolConfig,
    )

    # Risk config with volatility-based stop loss
    risk_config = RiskConfigModel(
        max_position_size=0.1,
        max_total_exposure=1.5,
        max_leverage=1.0,
        max_drawdown=0.25,
        max_daily_loss=0.05,
        stop_loss_enabled=True,
        stop_loss_pct=None,  # Use volatility-based stop instead
        take_profit_pct=0.15,
        stop_loss_params={
            "volatility_multiplier": 2.0,  # 2x rolling volatility stop
        },
    )

    config = BacktestConfig(
        mode=BacktestMode.CROSS_SECTIONAL,
        initial_capital=capital,
        start_date=timestamps_min,
        end_date=timestamps_max,
        rebalance_frequency=rebalance_frequency,
        top_n_long=top_n_long,
        top_n_short=top_n_short,
        ranking_factor=factor,
        risk_config=risk_config,
        cost_config=CostConfigModel(taker_rate=0.0005, slippage_rate=0.0003),
        asset_pool_config=BtAssetPoolConfig(
            enabled=True,
            reselect_frequency=pool_reselect,
            lookback_period=pool_lookback,
            selection_criteria=["dollar_volume"],
            top_n=pool_top_n,
        ),
        log_trades=True,
        log_signals=False,
        log_snapshots=False,
    )

    strategy = create_strategy(
        strategy_type,
        lookback_col=factor,
        top_n_long=top_n_long,
        top_n_short=top_n_short,
    )
    trading_logger = TradingLogger(
        task_id=f"dollarbar_{factor}", log_signals=False, log_snapshots=False,
    )
    engine = create_backtest_engine(config, strategy, logger=trading_logger)
    return engine


def run_segmented_backtest(
    symbol_files: Dict[str, List[Path]],
    segments: List[Dict],
    strategies: Dict[str, dict],
    feature_windows: List[int],
    initial_capital: float,
    top_n_long: int,
    top_n_short: int,
    rebalance_frequency: str,
    pool_top_n: int,
    pool_reselect: str,
    pool_lookback: str,
    profiles_dir: Optional[str] = None,
    max_symbols: int = 0,
) -> Dict[str, dict]:
    """Run segment-chained backtests for each strategy.

    For each strategy:
      - Iterate over time segments
      - Load data for that segment (+ warmup overlap)
      - Optionally filter by asset pool from profiles
      - Calculate features
      - Run backtest with capital carried over from previous segment
      - Collect NAV history and trades across all segments

    Returns:
        Dict mapping strategy_name -> performance metrics.
    """
    # Per-strategy accumulators
    strategy_nav: Dict[str, List[float]] = {}
    strategy_trades: Dict[str, list] = {}
    strategy_capital: Dict[str, float] = {}

    for segment_index, segment in enumerate(segments):
        segment_label = (
            f"{segment['eval_months'][0][0]}-{segment['eval_months'][0][1]:02d} "
            f"to {segment['eval_months'][-1][0]}-{segment['eval_months'][-1][1]:02d}"
        )
        print(f"\n{'='*70}")
        print(f"  Segment {segment_index + 1}/{len(segments)}: {segment_label}")
        print(f"  Loading {len(segment['load_months'])} months "
              f"(eval: {len(segment['eval_months'])}, "
              f"warmup: {len(segment['load_months']) - len(segment['eval_months'])})")
        print(f"{'='*70}")

        # --- Get asset pool from profiles for this segment ---
        asset_pool = None
        if profiles_dir:
            target_month = segment["eval_months"][0]
            asset_pool = load_asset_pool_from_profiles(
                profiles_dir, target_month, top_n=pool_top_n
            )
            if asset_pool:
                print(f"  Asset pool: {len(asset_pool)} symbols from profiles")

        # --- Load data for this segment ---
        step_start = time.time()
        bar_data = load_bar_data_for_months(
            symbol_files, segment["load_months"],
            asset_pool=asset_pool,
            max_symbols=max_symbols,
        )
        if not bar_data:
            logger.warning(f"  No data for segment {segment_label}, skipping")
            continue
        load_time = time.time() - step_start
        print(f"  Loaded {len(bar_data)} symbols ({load_time:.1f}s)")

        # --- Calculate features, trimming warmup rows ---
        step_start = time.time()
        eval_start_year, eval_start_month = segment["eval_months"][0]
        eval_start_ts = pd.Timestamp(
            year=eval_start_year, month=eval_start_month, day=1, tz="UTC"
        )
        try:
            featured_data = calculate_features(
                bar_data, feature_windows, eval_start=eval_start_ts,
            )
        except RuntimeError as error:
            logger.warning(f"  Feature calc failed for segment: {error}")
            del bar_data
            gc.collect()
            continue
        feat_time = time.time() - step_start
        print(f"  Features: {featured_data.shape[0]:,} rows ({feat_time:.1f}s)")

        # Free raw bar data
        del bar_data
        gc.collect()

        # --- Run backtest for each strategy on this segment ---
        available_cols = set(featured_data.columns)
        timestamps = featured_data.index.get_level_values(0)

        for strategy_name, strategy_config in strategies.items():
            factor = strategy_config["factor"]
            strategy_type = strategy_config["type"]

            if factor not in available_cols:
                logger.debug(f"  {strategy_name}: factor {factor} not in data")
                continue
            if featured_data[factor].notna().mean() < 0.1:
                logger.debug(f"  {strategy_name}: factor {factor} has too many NaN")
                continue

            # Determine starting capital for this segment
            capital = strategy_capital.get(strategy_name, initial_capital)

            engine = _create_engine_for_factor(
                factor=factor,
                capital=capital,
                timestamps_min=timestamps.min(),
                timestamps_max=timestamps.max(),
                rebalance_frequency=rebalance_frequency,
                top_n_long=top_n_long,
                top_n_short=top_n_short,
                pool_top_n=pool_top_n,
                pool_reselect=pool_reselect,
                pool_lookback=pool_lookback,
                strategy_type=strategy_type,
            )

            try:
                engine.run(featured_data)

                # Collect NAV
                nav_dict = engine.get_nav_history()
                nav_values = list(nav_dict.values()) if isinstance(nav_dict, dict) else []
                if nav_values:
                    strategy_nav.setdefault(strategy_name, []).extend(nav_values)
                    # Carry ending NAV as capital for next segment
                    strategy_capital[strategy_name] = nav_values[-1]

                # Collect trades
                trades = engine.get_trades()
                strategy_trades.setdefault(strategy_name, []).extend(trades)

            except Exception as error:
                logger.error(f"  {strategy_name} segment {segment_index+1} failed: {error}")

            del engine
            gc.collect()

        # Free featured data before next segment
        del featured_data
        gc.collect()

    # --- Compute final metrics ---
    results: Dict[str, dict] = {}
    for strategy_name in strategies.keys():
        nav_values = strategy_nav.get(strategy_name, [])
        trades = strategy_trades.get(strategy_name, [])

        if len(nav_values) < 2:
            logger.warning(f"  {strategy_name}: insufficient NAV points across segments")
            continue

        metrics = _calculate_metrics(nav_values, trades)
        results[strategy_name] = metrics

        print(
            f"\n  --- {strategy_name} ---"
            f"\n      Sharpe={metrics['sharpe_ratio']:.4f}  "
            f"Return={metrics['total_return_pct']:.2f}%  "
            f"MaxDD={metrics['max_drawdown_pct']:.2f}%  "
            f"Trades={metrics['total_trades']}"
        )

    return results


def _calculate_metrics(nav_values: List[float], trades) -> dict:
    """Compute performance metrics from NAV series and trades."""
    initial = nav_values[0]
    final = nav_values[-1]
    total_return = (final / initial - 1) * 100

    # Max drawdown
    peak = initial
    max_drawdown = 0.0
    for nav in nav_values:
        if nav > peak:
            peak = nav
        drawdown = (peak - nav) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Annualized metrics
    # For dollar bars, assume ~100 bars/day average, 252 trading days
    returns = np.diff(nav_values) / np.array(nav_values[:-1])
    bars_per_year = 100 * 252
    annualized_volatility = np.std(returns) * np.sqrt(bars_per_year)
    annualized_return = np.mean(returns) * bars_per_year
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

    winning_trades = sum(1 for t in trades if (getattr(t, "pnl", 0) or 0) > 0)
    win_rate = winning_trades / len(trades) * 100 if trades else 0

    return {
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "annualized_vol_pct": round(annualized_volatility * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_pipeline(
    bar_dir: str,
    profiles_dir: Optional[str] = None,
    max_symbols: int = 5,
) -> bool:
    """Run quick verification of the data pipeline.

    1. Load 1-2 months of data for a few symbols
    2. Calculate features
    3. Print diagnostic info
    4. Return True/False
    """
    print("\n" + "=" * 70)
    print("PIPELINE VERIFICATION")
    print("=" * 70)

    # Step 1: Scan and load
    print("\n[1] Scanning bar directory...")
    symbol_files = scan_available_months(bar_dir, min_months=2)
    if not symbol_files:
        print("  FAIL: No symbols found")
        return False
    print(f"  Found {len(symbol_files)} symbols")

    all_months = get_global_month_range(symbol_files)
    if len(all_months) < 2:
        print("  FAIL: Need at least 2 months of data")
        return False
    print(f"  Month range: {all_months[0]} to {all_months[-1]}")

    # Load first 2 months for max_symbols
    months_to_load = all_months[:2]
    print(f"\n[2] Loading {months_to_load} for {max_symbols} symbols...")

    # Get asset pool from profiles if available
    asset_pool = None
    if profiles_dir and Path(profiles_dir).exists():
        asset_pool = load_asset_pool_from_profiles(
            profiles_dir, months_to_load[0], top_n=max_symbols * 2
        )
        print(f"  Asset pool from profiles: {len(asset_pool) if asset_pool else 0} symbols")

    bar_data = load_bar_data_for_months(
        symbol_files, months_to_load,
        asset_pool=asset_pool,
        max_symbols=max_symbols,
    )
    if not bar_data:
        print("  FAIL: No bar data loaded")
        return False
    print(f"  Loaded {len(bar_data)} symbols")

    for symbol, df in list(bar_data.items())[:2]:
        print(f"    {symbol}: {len(df):,} rows, cols={list(df.columns)[:10]}...")

    # Step 2: Calculate features
    print("\n[3] Calculating features...")
    try:
        featured = calculate_features(bar_data, [5, 10, 20, 60])
    except Exception as e:
        print(f"  FAIL: Feature calculation error: {e}")
        return False

    print(f"  Shape: {featured.shape}")
    print(f"  Assets: {featured.index.get_level_values('asset').nunique()}")

    # Check key factors
    key_factors = ["momentum_20", "return_5", "signed_pe_20", "VPIN_50", "OFI_approx_20"]
    print("\n[4] Factor statistics:")
    for factor in key_factors:
        if factor in featured.columns:
            s = featured[factor]
            non_null = s.notna().sum()
            print(f"    {factor}: non_null={non_null:,}, mean={s.mean():.6f}, std={s.std():.6f}")
        else:
            print(f"    {factor}: NOT FOUND")

    # Step 3: Check forward return correlation
    print("\n[5] Predictive power (Spearman corr vs forward 20-bar return):")
    if "close" in featured.columns:
        # Reset index to work with it
        df_flat = featured.reset_index()
        df_flat["fwd_ret_20"] = df_flat.groupby("asset")["close"].transform(
            lambda x: x.shift(-20) / x - 1
        )
        for factor in key_factors:
            if factor in df_flat.columns:
                valid = df_flat[[factor, "fwd_ret_20"]].dropna()
                if len(valid) > 100:
                    corr = valid[factor].corr(valid["fwd_ret_20"], method="spearman")
                    print(f"    {factor}: corr={corr:.4f}, n={len(valid):,}")

    print("\n" + "=" * 70)
    print("VERIFICATION PASSED")
    print("=" * 70)
    return True


# ---------------------------------------------------------------------------
# Results Output
# ---------------------------------------------------------------------------

def print_results(results: Dict[str, dict], output_dir: str) -> None:
    """Print strategy comparison table and save results."""
    if not results:
        print("\n  [WARN] No results to display")
        return

    print("\n" + "=" * 95)
    print("STRATEGY COMPARISON  (cross-sectional, dollar bars, dynamic pool)")
    print("=" * 95)

    header = (
        f"  {'Strategy':<20} | {'Return%':>9} | {'MaxDD%':>8} | {'AnnRet%':>9} | "
        f"{'AnnVol%':>9} | {'Sharpe':>8} | {'Trades':>7} | {'WinR%':>6}"
    )
    print(header)
    print("  " + "-" * 91)

    sorted_strategies = sorted(
        results.items(), key=lambda x: x[1].get("sharpe_ratio", 0), reverse=True,
    )
    for strategy_name, metrics in sorted_strategies:
        print(
            f"  {strategy_name:<20} | "
            f"{metrics['total_return_pct']:>8.2f}% | "
            f"{metrics['max_drawdown_pct']:>7.2f}% | "
            f"{metrics['annualized_return_pct']:>8.2f}% | "
            f"{metrics['annualized_vol_pct']:>8.2f}% | "
            f"{metrics['sharpe_ratio']:>8.4f} | "
            f"{metrics['total_trades']:>7d} | "
            f"{metrics['win_rate_pct']:>5.1f}%"
        )

    print("  " + "-" * 91)
    if sorted_strategies:
        best_strategy = sorted_strategies[0][0]
        print(f"\n  Best by Sharpe: {best_strategy}")

    # Save JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / "dollarbar_comparison.json"
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Results saved to {result_file}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional backtest with dollar bars and dynamic asset pool"
    )

    # Data
    parser.add_argument(
        "--bar-dir", default="E:/data/dollar_bar/bars",
        help="Directory containing dollar bar parquet files",
    )
    parser.add_argument(
        "--profiles-dir", default="E:/data/dollar_bar/profiles",
        help="Directory containing daily profile parquet files for asset pool selection",
    )
    parser.add_argument(
        "--output-dir", default="E:/data/backtest_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--min-months", type=int, default=6,
        help="Minimum months of data required per symbol",
    )
    parser.add_argument(
        "--max-symbols", type=int, default=0,
        help="Max symbols to load (0 = all)",
    )

    # Backtest parameters
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--long-n", type=int, default=10)
    parser.add_argument("--short-n", type=int, default=10)
    parser.add_argument(
        "--rebalance", default="W-MON",
        help="Rebalance frequency: D, W-MON, W-FRI, MS, QS",
    )

    # Dynamic asset pool
    parser.add_argument(
        "--pool-top-n", type=int, default=100,
        help="Number of assets in the pool per period",
    )
    parser.add_argument(
        "--pool-reselect", default="MS",
        help="Pool refresh frequency: MS=monthly, QS=quarterly",
    )
    parser.add_argument(
        "--pool-lookback", default="30D",
        help="Lookback period for pool selection: 30D, 60D, 90D",
    )

    # Feature windows
    parser.add_argument(
        "--windows", nargs="+", type=int, default=[5, 10, 20, 60],
        help="Feature calculation windows",
    )

    # Segmented backtest (memory optimization)
    parser.add_argument(
        "--segment-months", type=int, default=6,
        help="Months per backtest segment (default 6). Smaller = less memory.",
    )
    parser.add_argument(
        "--warmup-months", type=int, default=2,
        help="Extra months loaded before each segment for feature warmup (default 2)",
    )

    # Strategy selection
    parser.add_argument(
        "--strategy", default=None,
        help=f"Run single strategy. Options: {list(STRATEGY_CONFIGS.keys())}. "
             "If not specified, runs all strategies.",
    )

    # Verification mode
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Run quick pipeline verification without full backtest",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    # Verification mode
    if args.verify_only:
        success = verify_pipeline(
            args.bar_dir,
            profiles_dir=args.profiles_dir,
            max_symbols=5,
        )
        sys.exit(0 if success else 1)

    print("=" * 70)
    print("DOLLAR BAR BACKTEST PIPELINE (segmented, memory-efficient)")
    print("=" * 70)
    print(f"  Bar dir:       {args.bar_dir}")
    print(f"  Profiles dir:  {args.profiles_dir}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Min months:    {args.min_months}")
    print(f"  Segment:       {args.segment_months} months + {args.warmup_months} warmup")
    print(f"  Pool:          top {args.pool_top_n}, "
          f"reselect={args.pool_reselect}, lookback={args.pool_lookback}")
    print(f"  Strategy:      L{args.long_n}/S{args.short_n}, "
          f"rebalance={args.rebalance}")
    print(f"  Windows:       {args.windows}")
    print()

    # Select strategies to run
    if args.strategy:
        if args.strategy not in STRATEGY_CONFIGS:
            print(f"[ERROR] Unknown strategy: {args.strategy}")
            print(f"Available: {list(STRATEGY_CONFIGS.keys())}")
            sys.exit(1)
        strategies = {args.strategy: STRATEGY_CONFIGS[args.strategy]}
    else:
        strategies = STRATEGY_CONFIGS

    print(f"  Strategies:    {list(strategies.keys())}")
    print()

    # Step 1: Scan available files (no data loaded)
    print("[Step 1] Scanning bar data directory...")
    step_start = time.time()
    symbol_files = scan_available_months(args.bar_dir, min_months=args.min_months)
    if not symbol_files:
        print("\n[FAIL] No bar data found.")
        sys.exit(1)

    all_months = get_global_month_range(symbol_files)
    segments = build_segments(
        all_months,
        segment_months=args.segment_months,
        warmup_months=args.warmup_months,
    )
    scan_time = time.time() - step_start
    print(f"  {len(symbol_files)} symbols, {len(all_months)} months total, "
          f"{len(segments)} segments ({scan_time:.1f}s)")

    # Step 2: Run segmented backtest
    print(f"\n[Step 2] Running segmented backtest...")

    comparison = run_segmented_backtest(
        symbol_files=symbol_files,
        segments=segments,
        strategies=strategies,
        feature_windows=args.windows,
        initial_capital=args.capital,
        top_n_long=args.long_n,
        top_n_short=args.short_n,
        rebalance_frequency=args.rebalance,
        pool_top_n=args.pool_top_n,
        pool_reselect=args.pool_reselect,
        pool_lookback=args.pool_lookback,
        profiles_dir=args.profiles_dir,
        max_symbols=args.max_symbols,
    )

    # Step 3: Report
    print_results(comparison, args.output_dir)

    elapsed = time.time() - total_start
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as error:
        logger.error(f"Pipeline failed: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
