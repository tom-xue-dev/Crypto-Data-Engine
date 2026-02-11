"""
Cross-sectional backtest using pre-aggregated time bar data.

Assumes time bars have already been generated via:
    main aggregate batch --bar-type time_bar --threshold 5min

Memory-efficient pipeline using **segmented backtesting**:
  - Splits the full time range into segments (default 6 months)
  - Each segment: load data → calculate features → run backtest → free memory
  - NAV and trades are chained across segments

Usage:
    python scripts/run_timebar_backtest.py
    python scripts/run_timebar_backtest.py --bar-dir E:/data/time_bar/bars
    python scripts/run_timebar_backtest.py --pool-top-n 50 --long-n 5 --short-n 5
    python scripts/run_timebar_backtest.py --rebalance W-MON --capital 1000000
    python scripts/run_timebar_backtest.py --segment-months 3 --warmup-months 2
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
logger = logging.getLogger("timebar_backtest")


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
        load_months  – months to load (includes warmup for feature calculation)
        eval_months  – months to actually evaluate in the backtest
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
# Step 1: Load bar data for a specific set of months
# ---------------------------------------------------------------------------

def load_bar_data_for_months(
    symbol_files: Dict[str, List[Path]],
    months_to_load: List[Tuple[int, int]],
    max_symbols: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Load bar data for specific months only, keeping memory low.

    Args:
        symbol_files: Pre-scanned symbol -> file list mapping.
        months_to_load: Which (year, month) files to load.
        max_symbols: Cap on number of symbols (0 = all).

    Returns:
        Dict mapping symbol -> merged DataFrame (only the requested months).
    """
    month_set = set(months_to_load)
    loaded: Dict[str, pd.DataFrame] = {}

    for symbol, files in symbol_files.items():
        if max_symbols > 0 and len(loaded) >= max_symbols:
            break

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

        loaded[symbol] = merged

    return loaded


# ---------------------------------------------------------------------------
# Step 2: Calculate features (per-symbol, memory-efficient)
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
            featured = calculator.calculate(bars, asset=asset)
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

    combined[time_col] = pd.to_datetime(combined[time_col], utc=True).dt.floor("5min")
    combined = combined.set_index([time_col, "asset"]).sort_index()

    asset_count = combined.index.get_level_values("asset").nunique()
    logger.info(
        f"Features: {combined.shape[0]:,} rows x {combined.shape[1]} cols, "
        f"{asset_count} assets (failed: {failed_count})"
    )
    return combined


# ---------------------------------------------------------------------------
# Step 3: Run backtest
# ---------------------------------------------------------------------------

# Factors that use mean-reversion logic: long losers (涨的少), short winners (涨的多)
REVERSAL_FACTORS = frozenset({"return_20", "return_60"})


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

    config = BacktestConfig(
        mode=BacktestMode.CROSS_SECTIONAL,
        initial_capital=capital,
        start_date=timestamps_min,
        end_date=timestamps_max,
        rebalance_frequency=rebalance_frequency,
        top_n_long=top_n_long,
        top_n_short=top_n_short,
        ranking_factor=factor,
        risk_config=RiskConfigModel(max_position_size=0.1, max_drawdown=0.3),
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

    # return_20, return_60: 做多涨的少的，做空涨的多的 (mean reversion)
    # others: 做多涨的多的，做空涨的少的 (momentum)
    strategy_name = "mean_reversion" if factor in REVERSAL_FACTORS else "momentum"
    strategy = create_strategy(
        strategy_name,
        lookback_col=factor,
        top_n_long=top_n_long,
        top_n_short=top_n_short,
    )
    trading_logger = TradingLogger(
        task_id=f"cmp_{factor}", log_signals=False, log_snapshots=False,
    )
    engine = create_backtest_engine(config, strategy, logger=trading_logger)
    return engine


def run_segmented_backtest(
    symbol_files: Dict[str, List[Path]],
    segments: List[Dict],
    factors: List[str],
    feature_windows: List[int],
    initial_capital: float,
    top_n_long: int,
    top_n_short: int,
    rebalance_frequency: str,
    pool_top_n: int,
    pool_reselect: str,
    pool_lookback: str,
    max_symbols: int = 0,
) -> Dict[str, dict]:
    """Run segment-chained backtests for each factor.

    For each factor:
      - Iterate over time segments
      - Load data for that segment (+ warmup overlap)
      - Calculate features
      - Run backtest with capital carried over from previous segment
      - Collect NAV history and trades across all segments

    Returns:
        Dict mapping factor -> performance metrics.
    """
    # Per-factor accumulators
    factor_nav: Dict[str, List[float]] = {}
    factor_trades: Dict[str, list] = {}
    factor_capital: Dict[str, float] = {}

    # Filter to valid factors
    valid_factors = list(factors)

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

        # --- Load data for this segment ---
        step_start = time.time()
        bar_data = load_bar_data_for_months(
            symbol_files, segment["load_months"], max_symbols=max_symbols,
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

        # --- Run backtest for each factor on this segment ---
        available_cols = set(featured_data.columns)
        timestamps = featured_data.index.get_level_values(0)

        for factor in valid_factors:
            if factor not in available_cols:
                continue
            if featured_data[factor].notna().mean() < 0.1:
                continue

            # Determine starting capital for this segment
            capital = factor_capital.get(factor, initial_capital)

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
            )

            try:
                engine.run(featured_data)

                # Collect NAV
                nav_dict = engine.get_nav_history()
                nav_values = list(nav_dict.values()) if isinstance(nav_dict, dict) else []
                if nav_values:
                    factor_nav.setdefault(factor, []).extend(nav_values)
                    # Carry ending NAV as capital for next segment
                    factor_capital[factor] = nav_values[-1]

                # Collect trades
                trades = engine.get_trades()
                factor_trades.setdefault(factor, []).extend(trades)

            except Exception as error:
                logger.error(f"  {factor} segment {segment_index+1} failed: {error}")

            del engine
            gc.collect()

        # Free featured data before next segment
        del featured_data
        gc.collect()

    # --- Compute final metrics ---
    results: Dict[str, dict] = {}
    for factor in valid_factors:
        nav_values = factor_nav.get(factor, [])
        trades = factor_trades.get(factor, [])

        if len(nav_values) < 2:
            logger.warning(f"  {factor}: insufficient NAV points across segments")
            continue

        metrics = _calculate_metrics(nav_values, trades)
        results[factor] = metrics

        print(
            f"\n  --- {factor} ---"
            f"\n      Sharpe={metrics['sharpe_ratio']:.4f}  "
            f"Return={metrics['total_return_pct']:.2f}%  "
            f"MaxDD={metrics['max_drawdown_pct']:.2f}%  "
            f"Trades={metrics['total_trades']}"
        )

    return results


def _extract_nav(engine) -> List[float]:
    """Get NAV history as a list."""
    nav = engine.get_nav_history()
    if isinstance(nav, dict):
        return list(nav.values())
    if isinstance(nav, pd.DataFrame):
        return nav.values.flatten().tolist()
    if isinstance(nav, pd.Series):
        return nav.tolist()
    return []


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

    # Annualized metrics (assume 5-min bars, ~288 bars/day, ~252 trading days)
    returns = np.diff(nav_values) / np.array(nav_values[:-1])
    annualized_volatility = np.std(returns) * np.sqrt(252 * 288)
    annualized_return = np.mean(returns) * 252 * 288
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
# Step 4: Report
# ---------------------------------------------------------------------------

def print_results(results: Dict[str, dict], output_dir: str) -> None:
    """Print factor comparison table and save results."""
    if not results:
        print("\n  [WARN] No results to display")
        return

    print("\n" + "=" * 95)
    print("FACTOR COMPARISON  (cross-sectional, dynamic pool, 5min time bars)")
    print("=" * 95)

    header = (
        f"  {'Factor':<28} | {'Return%':>9} | {'MaxDD%':>8} | {'AnnRet%':>9} | "
        f"{'AnnVol%':>9} | {'Sharpe':>8} | {'Trades':>7} | {'WinR%':>6}"
    )
    print(header)
    print("  " + "-" * 91)

    sorted_factors = sorted(
        results.items(), key=lambda x: x[1].get("sharpe_ratio", 0), reverse=True,
    )
    for factor, metrics in sorted_factors:
        print(
            f"  {factor:<28} | "
            f"{metrics['total_return_pct']:>8.2f}% | "
            f"{metrics['max_drawdown_pct']:>7.2f}% | "
            f"{metrics['annualized_return_pct']:>8.2f}% | "
            f"{metrics['annualized_vol_pct']:>8.2f}% | "
            f"{metrics['sharpe_ratio']:>8.4f} | "
            f"{metrics['total_trades']:>7d} | "
            f"{metrics['win_rate_pct']:>5.1f}%"
        )

    print("  " + "-" * 91)
    best_factor = sorted_factors[0][0]
    print(f"\n  Best by Sharpe: {best_factor}")

    # Save JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / "factor_comparison.json"
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Results saved to {result_file}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional backtest with pre-aggregated time bars"
    )

    # Data
    parser.add_argument(
        "--bar-dir", default="E:/data/time_bar/bars",
        help="Directory containing time bar parquet files",
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

    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    print("=" * 70)
    print("BACKTEST PIPELINE (segmented, memory-efficient)")
    print("=" * 70)
    print(f"  Bar dir:       {args.bar_dir}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Min months:    {args.min_months}")
    print(f"  Segment:       {args.segment_months} months + {args.warmup_months} warmup")
    print(f"  Pool:          top {args.pool_top_n}, "
          f"reselect={args.pool_reselect}, lookback={args.pool_lookback}")
    print(f"  Strategy:      momentum L{args.long_n}/S{args.short_n}, "
          f"rebalance={args.rebalance}")
    print(f"  Windows:       {args.windows}")
    print()

    # Step 1: Scan available files (no data loaded)
    print("[Step 1] Scanning bar data directory...")
    step_start = time.time()
    symbol_files = scan_available_months(args.bar_dir, min_months=args.min_months)
    if not symbol_files:
        print("\n[FAIL] No bar data found. Run aggregation first:")
        print("  main aggregate batch --bar-type time_bar --threshold 5min")
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

    # Step 2 + 3: Segmented backtest (load → features → backtest per segment)
    print(f"\n[Step 2] Running segmented backtest...")

    factors = [
        "return_20", "return_60",
        "momentum_20",
        "path_efficiency_20",
        "signed_pe_5", "signed_pe_10", "signed_pe_20",
        "impact_density_5", "impact_density_10", "impact_density_20",
    ]

    comparison = run_segmented_backtest(
        symbol_files=symbol_files,
        segments=segments,
        factors=factors,
        feature_windows=args.windows,
        initial_capital=args.capital,
        top_n_long=args.long_n,
        top_n_short=args.short_n,
        rebalance_frequency=args.rebalance,
        pool_top_n=args.pool_top_n,
        pool_reselect=args.pool_reselect,
        pool_lookback=args.pool_lookback,
        max_symbols=args.max_symbols,
    )

    # Step 4: Report
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
