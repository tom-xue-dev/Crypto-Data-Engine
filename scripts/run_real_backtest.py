"""
End-to-end real data backtest with dynamic (historical) asset pool.

Pipeline:
  1. Scan disk for all symbols with tick data
  2. Aggregate tick -> 1h bars (streaming, per-file)
  3. Calculate features (returns, volatility, momentum, PE, IID, ...)
  4. Run cross-sectional backtest with monthly-refreshed asset pool
     (month X pool = top N by month X-1 avg dollar_volume)
  5. Compare multiple ranking factors side by side

Usage:
    python scripts/run_real_backtest.py
    python scripts/run_real_backtest.py --skip-aggregation
    python scripts/run_real_backtest.py --pool-top-n 50 --long-n 5 --short-n 5
    python scripts/run_real_backtest.py --pool-reselect QS --pool-lookback 90D
"""
import argparse
import gc
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
)
logger = logging.getLogger("real_backtest")


# ---------------------------------------------------------------------------
# Tick loading (module-level for Windows multiprocessing pickling)
# ---------------------------------------------------------------------------

_TICK_COLUMNS = ["transact_time", "price", "quantity", "isBuyerMaker"]


def _load_and_prepare_chunk(parquet_file: Path) -> Optional[pd.DataFrame]:
    """Load one parquet file, rename columns, downcast to float32."""
    try:
        chunk = pd.read_parquet(str(parquet_file), columns=_TICK_COLUMNS)
    except Exception:
        chunk = pd.read_parquet(str(parquet_file))

    if chunk.empty:
        return None

    chunk = chunk.rename(columns={"transact_time": "timestamp"})

    if not {"timestamp", "price", "quantity"}.issubset(chunk.columns):
        return None

    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").astype(np.float32)
    chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="coerce").astype(np.float32)
    chunk = chunk.dropna(subset=["price", "quantity"])

    if "isBuyerMaker" not in chunk.columns:
        chunk["isBuyerMaker"] = False

    return chunk


def _aggregate_single_asset(
    args: Tuple[str, str, str, str, bool],
) -> Optional[Tuple[str, str]]:
    """Aggregate tick data to 1h bars for one symbol (subprocess-safe).

    Streams parquet files one at a time to keep memory usage low.
    """
    asset, tick_data_dir, bar_output_dir, bar_threshold, include_advanced = args
    try:
        from crypto_data_engine.services.bar_aggregator import aggregate_bars

        tick_dir = Path(tick_data_dir) / asset
        if not tick_dir.exists():
            return None

        parquet_files = sorted(tick_dir.rglob("*.parquet"))
        if not parquet_files:
            return None

        all_bar_frames: List[pd.DataFrame] = []
        total_ticks = 0

        for parquet_file in parquet_files:
            chunk = _load_and_prepare_chunk(parquet_file)
            if chunk is None or len(chunk) == 0:
                continue
            total_ticks += len(chunk)

            bars = aggregate_bars(
                chunk,
                bar_type="time_bar",
                threshold=bar_threshold,
                use_numba=True,
                include_advanced=include_advanced,
                symbol=asset,
            )
            if isinstance(bars, pd.DataFrame) and len(bars) > 0:
                all_bar_frames.append(bars)
            del chunk

        if not all_bar_frames or total_ticks < 100:
            return None

        combined_bars = pd.concat(all_bar_frames, ignore_index=True)
        del all_bar_frames

        output_dir = Path(bar_output_dir) / asset
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{asset}_time_bar_{bar_threshold}.parquet"
        combined_bars.to_parquet(str(output_path), index=False)
        return (asset, str(output_path))

    except Exception as error:
        print(f"  [ERROR] {asset}: {error}")
        return None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_scan_symbols(tick_data_dir: str) -> List[str]:
    """Step 1: Scan disk for all symbols that have tick parquet data."""
    tick_dir = Path(tick_data_dir)
    symbols = sorted([
        directory.name
        for directory in tick_dir.iterdir()
        if directory.is_dir() and any(directory.rglob("*.parquet"))
    ])
    logger.info(f"Found {len(symbols)} symbols with tick data on disk")
    return symbols


def _safe_worker_count(requested: int, per_worker_gb: float = 1.5) -> int:
    """Cap workers to available memory."""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        return max(1, min(requested, int(available_gb / per_worker_gb)))
    except ImportError:
        return requested


def step_aggregate_bars(
    symbols: List[str],
    tick_data_dir: str,
    bar_output_dir: str,
    bar_threshold: str = "1h",
    max_workers: int = 4,
) -> Dict[str, str]:
    """Step 2: Aggregate tick → bars in parallel. Returns {asset: path}."""
    Path(bar_output_dir).mkdir(parents=True, exist_ok=True)

    available = [
        symbol for symbol in symbols
        if (Path(tick_data_dir) / symbol).exists()
        and any((Path(tick_data_dir) / symbol).rglob("*.parquet"))
    ]
    if not available:
        logger.warning("No tick data found")
        return {}

    effective_workers = _safe_worker_count(max_workers)
    logger.info(f"Aggregating {len(available)} symbols, workers={effective_workers}")

    task_args = [
        (symbol, tick_data_dir, bar_output_dir, bar_threshold, True)
        for symbol in available
    ]

    results: Dict[str, str] = {}
    done = 0

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(_aggregate_single_asset, a): a[0] for a in task_args
        }
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                if result:
                    results[result[0]] = result[1]
                if done % 20 == 0 or done == len(futures):
                    logger.info(f"  Progress: {done}/{len(futures)} ({len(results)} ok)")
            except Exception as error:
                logger.warning(f"  {futures[future]}: {error}")

    logger.info(f"Aggregation done: {len(results)}/{len(available)}")
    return results


def step_load_existing_bars(bar_dir: str) -> Dict[str, str]:
    """Step 2 (skip): Load all existing bar parquet files from disk."""
    bar_path = Path(bar_dir)
    if not bar_path.exists():
        return {}
    bar_paths: Dict[str, str] = {}
    for symbol_dir in sorted(bar_path.iterdir()):
        if symbol_dir.is_dir():
            parquets = sorted(symbol_dir.glob("*.parquet"))
            if parquets:
                bar_paths[symbol_dir.name] = str(parquets[0])
    return bar_paths


def step_calculate_features(
    bar_paths: Dict[str, str],
    feature_windows: List[int],
) -> pd.DataFrame:
    """Step 3: Calculate features → MultiIndex DataFrame (timestamp, asset)."""
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
    for asset, bar_path in bar_paths.items():
        try:
            bars = pd.read_parquet(bar_path)
            if len(bars) < 30:
                continue
            featured = calculator.calculate(bars, asset=asset)
            featured["asset"] = asset
            frames.append(featured)
        except Exception as error:
            logger.warning(f"  {asset}: features failed - {error}")

    if not frames:
        raise RuntimeError("No assets with valid features")

    combined = pd.concat(frames, ignore_index=True)

    time_col = next(
        (c for c in ["start_time", "timestamp", "time"] if c in combined.columns),
        None,
    )
    if time_col is None:
        raise RuntimeError(f"No time column. Columns: {list(combined.columns)}")

    combined[time_col] = pd.to_datetime(combined[time_col]).dt.floor("h")
    combined = combined.set_index([time_col, "asset"]).sort_index()

    logger.info(
        f"Features: {combined.shape[0]} rows × {combined.shape[1]} cols, "
        f"{combined.index.get_level_values('asset').nunique()} assets"
    )
    return combined


def _build_backtest_config(
    data: pd.DataFrame,
    ranking_factor: str,
    top_n_long: int,
    top_n_short: int,
    rebalance_frequency: str,
    initial_capital: float,
    pool_top_n: int,
    pool_reselect: str,
    pool_lookback: str,
    log_signals: bool = False,
):
    """Construct a BacktestConfig with dynamic asset pool enabled."""
    from crypto_data_engine.services.back_test import (
        BacktestConfig,
        BacktestMode,
        CostConfigModel,
        RiskConfigModel,
    )
    from crypto_data_engine.services.back_test.config import (
        AssetPoolConfig as BtAssetPoolConfig,
    )

    timestamps = data.index.get_level_values(0)

    return BacktestConfig(
        mode=BacktestMode.CROSS_SECTIONAL,
        initial_capital=initial_capital,
        start_date=timestamps.min(),
        end_date=timestamps.max(),
        rebalance_frequency=rebalance_frequency,
        top_n_long=top_n_long,
        top_n_short=top_n_short,
        ranking_factor=ranking_factor,
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
        log_signals=log_signals,
        log_snapshots=False,
    )


def _extract_nav(engine) -> List[float]:
    """Get NAV values list from engine, handling dict/Series/DataFrame."""
    nav = engine.get_nav_history()
    if isinstance(nav, dict):
        return list(nav.values())
    if isinstance(nav, pd.DataFrame):
        return nav.values.flatten().tolist()
    if isinstance(nav, pd.Series):
        return nav.tolist()
    return []


def _calc_metrics(nav_values: List[float], trades) -> dict:
    """Compute Sharpe, return, drawdown, win rate from NAV + trades."""
    initial = nav_values[0]
    final = nav_values[-1]
    total_return = (final / initial - 1) * 100

    peak = initial
    max_dd = 0.0
    for nav in nav_values:
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak
        if dd > max_dd:
            max_dd = dd

    returns = np.diff(nav_values) / np.array(nav_values[:-1])
    ann_vol = np.std(returns) * np.sqrt(252 * 24)
    ann_ret = np.mean(returns) * 252 * 24
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    winning = sum(1 for t in trades if (getattr(t, "pnl", 0) or 0) > 0)
    win_rate = winning / len(trades) * 100 if trades else 0

    return {
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "annualized_vol_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
    }


def step_compare_factors(
    data: pd.DataFrame,
    factors: List[str],
    initial_capital: float,
    top_n_long: int,
    top_n_short: int,
    rebalance_frequency: str,
    pool_top_n: int,
    pool_reselect: str,
    pool_lookback: str,
) -> Dict[str, dict]:
    """Step 4: Run backtest with each factor, collect metrics."""
    from crypto_data_engine.services.back_test import (
        TradingLogger,
        create_backtest_engine,
        create_strategy,
    )

    available_cols = set(data.columns)
    results: Dict[str, dict] = {}

    for factor in factors:
        if factor not in available_cols:
            logger.warning(f"Factor '{factor}' not in data, skip")
            continue
        if data[factor].notna().mean() < 0.3:
            logger.warning(f"Factor '{factor}' too few non-NaN, skip")
            continue

        print(f"\n  --- {factor} ---")

        config = _build_backtest_config(
            data, factor, top_n_long, top_n_short,
            rebalance_frequency, initial_capital,
            pool_top_n, pool_reselect, pool_lookback,
        )
        strategy = create_strategy(
            "momentum", lookback_col=factor,
            top_n_long=top_n_long, top_n_short=top_n_short,
        )
        trading_logger = TradingLogger(
            task_id=f"cmp_{factor}", log_signals=False, log_snapshots=False,
        )

        try:
            engine = create_backtest_engine(config, strategy, logger=trading_logger)
            engine.run(data)

            nav_values = _extract_nav(engine)
            if len(nav_values) < 2:
                logger.warning(f"  {factor}: < 2 NAV points, skip")
                continue

            trades = engine.get_trades()  # aggregated by default
            metrics = _calc_metrics(nav_values, trades)
            metrics["_trades"] = trades  # stash for later trade log saving
            results[factor] = metrics
            print(
                f"      Sharpe={metrics['sharpe_ratio']:.4f}  "
                f"Return={metrics['total_return_pct']:.2f}%  "
                f"MaxDD={metrics['max_drawdown_pct']:.2f}%  "
                f"Trades={metrics['total_trades']}"
            )
        except Exception as error:
            logger.error(f"  {factor} failed: {error}")
            import traceback
            traceback.print_exc()

    return results


def print_comparison(results: Dict[str, dict], output_dir: str) -> None:
    """Step 5: Print factor comparison table, save JSON."""
    if not results:
        print("\n  [WARN] No results")
        return

    print("\n" + "=" * 92)
    print("FACTOR COMPARISON  (dynamic asset pool, month X pool = top by month X-1 volume)")
    print("=" * 92)

    header = (
        f"  {'Factor':<28} | {'Return%':>9} | {'MaxDD%':>8} | {'AnnRet%':>9} | "
        f"{'AnnVol%':>9} | {'Sharpe':>8} | {'Trades':>7} | {'WinR%':>6}"
    )
    print(header)
    print("  " + "-" * 88)

    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1].get("sharpe_ratio", 0),
        reverse=True,
    )
    for factor, m in sorted_items:
        print(
            f"  {factor:<28} | "
            f"{m['total_return_pct']:>8.2f}% | "
            f"{m['max_drawdown_pct']:>7.2f}% | "
            f"{m['annualized_return_pct']:>8.2f}% | "
            f"{m['annualized_vol_pct']:>8.2f}% | "
            f"{m['sharpe_ratio']:>8.4f} | "
            f"{m['total_trades']:>7d} | "
            f"{m['win_rate_pct']:>5.1f}%"
        )

    print("  " + "-" * 88)

    best_factor = sorted_items[0][0]
    print(f"\n  Best by Sharpe: {best_factor}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save comparison JSON (without internal _trades lists)
    save_data = {
        factor: {k: v for k, v in m.items() if not k.startswith("_")}
        for factor, m in results.items()
    }
    with open(out / "factor_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(save_data, fh, indent=2)

    # Save trade log for the best factor
    best_trades = results[best_factor].get("_trades", [])
    if best_trades:
        trade_records = []
        for trade in best_trades:
            direction = getattr(trade, "direction", "?")
            if hasattr(direction, "value"):
                direction = direction.value
            trade_records.append({
                "asset": str(getattr(trade, "asset", "?")),
                "direction": str(direction),
                "entry_time": str(getattr(trade, "entry_time", ""))[:19],
                "exit_time": str(getattr(trade, "exit_time", ""))[:19],
                "entry_price": round(getattr(trade, "entry_price", 0), 6),
                "exit_price": round(getattr(trade, "exit_price", 0) or 0, 6),
                "quantity": round(getattr(trade, "quantity", 0), 6),
                "pnl": round(getattr(trade, "pnl", 0), 2),
                "pnl_pct": round((getattr(trade, "pnl_pct", 0) or 0) * 100, 4),
            })
        with open(out / "trade_log.json", "w", encoding="utf-8") as fh:
            json.dump(trade_records, fh, indent=2, ensure_ascii=False)
        logger.info(f"Trade log ({best_factor}, {len(trade_records)} trades) -> {out / 'trade_log.json'}")

    logger.info(f"Comparison -> {out / 'factor_comparison.json'}")
    print("=" * 92)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest with dynamic asset pool (historical volume)"
    )
    parser.add_argument("--data-dir", default="E:/data/binance_futures")
    parser.add_argument("--bar-dir", default="E:/data/bar_data")
    parser.add_argument("--output-dir", default="E:/data/backtest_results")
    parser.add_argument("--threshold", default="1h", help="Bar interval")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-aggregation", action="store_true")
    # Backtest
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--long-n", type=int, default=10)
    parser.add_argument("--short-n", type=int, default=10)
    parser.add_argument("--rebalance", default="W-MON")
    # Dynamic pool (always on)
    parser.add_argument("--pool-top-n", type=int, default=100,
                        help="Assets per period pool")
    parser.add_argument("--pool-reselect", default="MS",
                        help="Pool refresh freq: MS=monthly, QS=quarterly")
    parser.add_argument("--pool-lookback", default="30D",
                        help="Lookback for avg dollar-volume: 30D, 90D")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("BACKTEST PIPELINE (dynamic historical asset pool)")
    print("=" * 70)
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Bar dir:       {args.bar_dir}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Threshold:     {args.threshold}")
    print(f"  Workers:       {args.workers}")
    print(f"  Pool:          top {args.pool_top_n}, "
          f"reselect={args.pool_reselect}, lookback={args.pool_lookback}")
    print(f"  Strategy:      momentum L{args.long_n}/S{args.short_n}, "
          f"rebalance={args.rebalance}")
    print()

    # Step 1: Scan symbols
    print("[Step 1] Scanning disk for symbols...")
    t = time.time()
    symbols = step_scan_symbols(args.data_dir)
    print(f"  Found {len(symbols)} symbols ({time.time() - t:.1f}s)")

    # Step 2: Aggregate bars
    if args.skip_aggregation:
        print("\n[Step 2] Loading existing bars...")
        bar_paths = step_load_existing_bars(args.bar_dir)
        print(f"  Loaded {len(bar_paths)} assets")
    else:
        print(f"\n[Step 2] Aggregating tick → {args.threshold} bars...")
        t = time.time()
        bar_paths = step_aggregate_bars(
            symbols, args.data_dir, args.bar_dir,
            args.threshold, args.workers,
        )
        print(f"  Aggregated {len(bar_paths)} assets ({time.time() - t:.1f}s)")
        gc.collect()

    if not bar_paths:
        print("\n[FAIL] No bar data. Download tick data first.")
        sys.exit(1)

    # Step 3: Features
    print(f"\n[Step 3] Calculating features for {len(bar_paths)} assets...")
    t = time.time()
    featured_data = step_calculate_features(bar_paths, [5, 10, 20, 60])
    print(f"  Shape: {featured_data.shape} ({time.time() - t:.1f}s)")
    gc.collect()

    # Step 4: Factor comparison backtest
    print("\n[Step 4] Running factor comparison backtest...")
    t = time.time()

    factors = [
        # Baseline momentum
        "return_20", "return_60", "momentum_20",
        # Unsigned Path Efficiency (directionless)
        "path_efficiency_20",
        # Signed Path Efficiency = PE × sign(return): positive = trending up
        "signed_pe_5", "signed_pe_10", "signed_pe_20",
        # Impact Density
        "impact_density_5", "impact_density_10", "impact_density_20",
    ]

    comparison = step_compare_factors(
        data=featured_data,
        factors=factors,
        initial_capital=args.capital,
        top_n_long=args.long_n,
        top_n_short=args.short_n,
        rebalance_frequency=args.rebalance,
        pool_top_n=args.pool_top_n,
        pool_reselect=args.pool_reselect,
        pool_lookback=args.pool_lookback,
    )
    print(f"  Completed ({time.time() - t:.1f}s)")

    # Step 5: Report
    print_comparison(comparison, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\nTotal: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


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
