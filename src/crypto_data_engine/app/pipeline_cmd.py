"""
Full pipeline CLI command.

Orchestrates: asset pool selection -> bar aggregation -> feature calculation
              -> cross-sectional backtest -> validation & report.

Core step functions are imported from scripts/run_real_backtest.py so that
the same logic can be invoked from either the CLI or directly as a script.
"""
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger("pipeline")

pipeline_app = typer.Typer(help="End-to-end pipeline (select -> aggregate -> features -> backtest -> validate)")


# ---------------------------------------------------------------------------
# Module-level function for multiprocessing (must be picklable on Windows)
# ---------------------------------------------------------------------------

# Columns we actually need from tick parquet files.
_TICK_COLUMNS = ["transact_time", "price", "quantity", "isBuyerMaker"]


def _load_and_prepare_chunk(parquet_file: Path) -> Optional[pd.DataFrame]:
    """Load a single parquet file, rename columns, downcast types.

    Returns None if the file is empty or lacks required columns.
    """
    try:
        chunk = pd.read_parquet(str(parquet_file), columns=_TICK_COLUMNS)
    except Exception:
        # Some files may lack certain columns – fall back to full read
        chunk = pd.read_parquet(str(parquet_file))

    if chunk.empty:
        return None

    chunk = chunk.rename(columns={"transact_time": "timestamp"})

    required = {"timestamp", "price", "quantity"}
    if not required.issubset(chunk.columns):
        return None

    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").astype(np.float32)
    chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="coerce").astype(np.float32)
    chunk = chunk.dropna(subset=["price", "quantity"])

    if "isBuyerMaker" not in chunk.columns:
        chunk["isBuyerMaker"] = False

    return chunk


def _aggregate_single_asset(args: Tuple[str, str, str, str, bool]) -> Optional[Tuple[str, str]]:
    """Aggregate tick data to bars for a single asset (subprocess-safe).

    Uses **streaming per-file aggregation** so that only one month of
    tick data is in memory at a time, instead of loading everything at
    once.  For time bars each file produces independent bars (no
    cross-file state needed apart from the boundary bar which is handled
    via leftover tick forwarding).
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

            # Explicit cleanup – this is the critical memory saving
            del chunk

        if not all_bar_frames or total_ticks < 100:
            return None

        combined_bars = pd.concat(all_bar_frames, ignore_index=True)
        del all_bar_frames  # free list immediately

        if len(combined_bars) == 0:
            return None

        output_dir = Path(bar_output_dir) / asset
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{asset}_time_bar_{bar_threshold}.parquet"
        combined_bars.to_parquet(str(output_path), index=False)
        return (asset, str(output_path))

    except Exception as error:
        print(f"  [ERROR] {asset}: aggregation failed - {error}")
        return None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_select_asset_pool(top_n: int, cache_dir: str = "E:/data") -> List[str]:
    """Select top N assets by trading volume from Binance Futures."""
    from crypto_data_engine.services.asset_pool.asset_selector import (
        AssetPoolConfig,
        AssetPoolSelector,
    )

    config = AssetPoolConfig(top_n=top_n, cache_dir=Path(cache_dir))
    selector = AssetPoolSelector(config)
    symbols = selector.get_current_pool(force_refresh=True)

    logger.info(f"Asset pool: {len(symbols)} symbols selected")
    return symbols


def _safe_worker_count(requested_workers: int, per_worker_gb: float = 1.5) -> int:
    """Return a worker count that fits in available memory.

    Falls back to ``requested_workers`` if *psutil* is not installed.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        safe = max(1, int(available_gb / per_worker_gb))
        return min(requested_workers, safe)
    except ImportError:
        return requested_workers


def step_aggregate_bars(
    symbols: List[str],
    tick_data_dir: str,
    bar_output_dir: str,
    bar_threshold: str = "1h",
    include_advanced: bool = True,
    max_workers: int = 4,
) -> Dict[str, str]:
    """Aggregate tick data to bars in parallel. Returns {asset: bar_path}."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    bar_dir = Path(bar_output_dir)
    bar_dir.mkdir(parents=True, exist_ok=True)

    available_symbols = [
        symbol for symbol in symbols
        if (Path(tick_data_dir) / symbol).exists()
        and any((Path(tick_data_dir) / symbol).rglob("*.parquet"))
    ]

    if not available_symbols:
        logger.warning("No tick data found for any symbol")
        return {}

    logger.info(f"Found tick data for {len(available_symbols)}/{len(symbols)} symbols")

    # Adaptive worker count: cap to available memory
    effective_workers = _safe_worker_count(max_workers)
    if effective_workers < max_workers:
        logger.info(
            f"Adaptive workers: reduced {max_workers} -> {effective_workers} "
            f"based on available memory"
        )

    task_args = [
        (symbol, tick_data_dir, bar_output_dir, bar_threshold, include_advanced)
        for symbol in available_symbols
    ]

    results: Dict[str, str] = {}
    completed_count = 0

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_aggregate_single_asset, args): args[0] for args in task_args}
        for future in as_completed(futures):
            completed_count += 1
            try:
                result = future.result()
                if result:
                    asset_name, output_path = result
                    results[asset_name] = output_path
                if completed_count % 10 == 0 or completed_count == len(futures):
                    logger.info(f"  Aggregation progress: {completed_count}/{len(futures)} ({len(results)} succeeded)")
            except Exception as error:
                logger.warning(f"  aggregation failed - {error}")

    logger.info(f"Aggregation complete: {len(results)}/{len(available_symbols)} assets")
    return results


def step_calculate_features(
    bar_paths: Dict[str, str],
    feature_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Calculate features for each asset and combine into MultiIndex DataFrame."""
    from crypto_data_engine.services.feature.unified_features import (
        UnifiedFeatureCalculator,
        UnifiedFeatureConfig,
    )

    windows = feature_windows or [5, 10, 20, 60]
    config = UnifiedFeatureConfig(
        windows=windows,
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

    all_featured: List[pd.DataFrame] = []
    for asset, bar_path in bar_paths.items():
        try:
            bars = pd.read_parquet(bar_path)
            if len(bars) < 30:
                continue
            featured = calculator.calculate(bars, asset=asset)
            featured["asset"] = asset
            all_featured.append(featured)
        except Exception as error:
            logger.warning(f"  {asset}: feature calculation failed - {error}")

    if not all_featured:
        raise RuntimeError("No assets with valid features after calculation")

    combined = pd.concat(all_featured, ignore_index=True)

    time_col = None
    for candidate in ["start_time", "timestamp", "time", "datetime"]:
        if candidate in combined.columns:
            time_col = candidate
            break

    if time_col is None:
        raise RuntimeError(f"No time column found. Columns: {list(combined.columns)}")

    combined[time_col] = pd.to_datetime(combined[time_col]).dt.floor("h")
    combined = combined.set_index([time_col, "asset"]).sort_index()

    logger.info(
        f"Features calculated: {combined.shape[0]} rows, "
        f"{combined.shape[1]} columns, "
        f"{combined.index.get_level_values('asset').nunique()} assets"
    )
    return combined


def step_run_backtest(
    data: pd.DataFrame,
    initial_capital: float = 1_000_000,
    top_n_long: int = 10,
    top_n_short: int = 10,
    rebalance_frequency: str = "W-MON",
    ranking_factor: str = "return_20",
    dynamic_pool: bool = False,
    pool_top_n: int = 100,
    pool_reselect_frequency: str = "MS",
    pool_lookback_period: str = "30D",
) -> dict:
    """Run cross-sectional momentum backtest.

    Args:
        dynamic_pool: Enable periodic asset-pool re-selection inside the
            backtest engine.  When True, the engine selects the top
            ``pool_top_n`` assets by rolling average dollar-volume at
            every ``pool_reselect_frequency`` interval, using a
            ``pool_lookback_period`` lookback window.
        pool_top_n: Number of assets to keep in each period's pool.
        pool_reselect_frequency: Pandas offset string for pool refresh
            (e.g. "MS" = month start, "QS" = quarter start).
        pool_lookback_period: Lookback window for computing average
            dollar-volume (e.g. "30D", "90D").
    """
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

    timestamps = data.index.get_level_values(0)
    start = timestamps.min()
    end = timestamps.max()

    logger.info(f"Backtest period: {start} -> {end}")
    logger.info(f"Strategy: momentum long={top_n_long} short={top_n_short}, factor={ranking_factor}")

    # Build asset-pool config
    asset_pool_cfg = BtAssetPoolConfig(
        enabled=dynamic_pool,
        reselect_frequency=pool_reselect_frequency,
        lookback_period=pool_lookback_period,
        selection_criteria=["dollar_volume"],
        top_n=pool_top_n,
    )
    if dynamic_pool:
        logger.info(
            f"Dynamic asset pool: top_n={pool_top_n}, "
            f"reselect={pool_reselect_frequency}, lookback={pool_lookback_period}"
        )

    config = BacktestConfig(
        mode=BacktestMode.CROSS_SECTIONAL,
        initial_capital=initial_capital,
        start_date=start,
        end_date=end,
        rebalance_frequency=rebalance_frequency,
        top_n_long=top_n_long,
        top_n_short=top_n_short,
        ranking_factor=ranking_factor,
        risk_config=RiskConfigModel(max_position_size=0.1, max_drawdown=0.3),
        cost_config=CostConfigModel(taker_rate=0.0005, slippage_rate=0.0003),
        asset_pool_config=asset_pool_cfg,
        log_trades=True,
        log_signals=True,
        log_snapshots=True,
    )

    strategy = create_strategy(
        "momentum",
        lookback_col=ranking_factor,
        top_n_long=top_n_long,
        top_n_short=top_n_short,
    )

    trading_logger = TradingLogger(
        task_id="pipeline_backtest",
        log_signals=True,
        log_snapshots=True,
        snapshot_frequency=1,
    )

    engine = create_backtest_engine(config, strategy, logger=trading_logger)
    result = engine.run(data)

    return {
        "result": result,
        "nav_history": engine.get_nav_history(),
        "trades": engine.get_trades(),
        "config": config,
        "logger": trading_logger,
    }


def step_validate_and_report(backtest_output: dict, output_dir: str) -> None:
    """Validate results, save trade log, and print metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nav_history = backtest_output["nav_history"]
    trades = backtest_output["trades"]

    if isinstance(nav_history, dict):
        nav_values = list(nav_history.values())
        nav_timestamps = list(nav_history.keys())
    elif isinstance(nav_history, (pd.DataFrame, pd.Series)):
        nav_values = nav_history.values.flatten().tolist() if isinstance(nav_history, pd.DataFrame) else nav_history.tolist()
        nav_timestamps = nav_history.index.tolist()
    else:
        nav_values, nav_timestamps = [], []

    typer.echo("\n" + "=" * 70)
    typer.echo("BACKTEST RESULT VALIDATION")
    typer.echo("=" * 70)

    if not nav_values:
        typer.echo("  [FAIL] No NAV data returned")
        return

    initial_nav = nav_values[0]
    final_nav = nav_values[-1]
    total_return_pct = (final_nav / initial_nav - 1) * 100

    peak = initial_nav
    max_drawdown = 0.0
    for nav in nav_values:
        if nav > peak:
            peak = nav
        drawdown = (peak - nav) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    returns = np.diff(nav_values) / np.array(nav_values[:-1])
    annualized_vol = np.std(returns) * np.sqrt(252 * 24) if len(returns) > 0 else 0
    annualized_return = np.mean(returns) * 252 * 24 if len(returns) > 0 else 0
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

    typer.echo("\n### NAV Summary")
    typer.echo(f"  Initial NAV:       {initial_nav:>15,.2f}")
    typer.echo(f"  Final NAV:         {final_nav:>15,.2f}")
    typer.echo(f"  Total Return:      {total_return_pct:>14.2f}%")
    typer.echo(f"  Max Drawdown:      {max_drawdown * 100:>14.2f}%")
    typer.echo(f"  Annualized Vol:    {annualized_vol * 100:>14.2f}%")
    typer.echo(f"  Annualized Return: {annualized_return * 100:>14.2f}%")
    typer.echo(f"  Sharpe Ratio:      {sharpe_ratio:>14.4f}")
    typer.echo(f"  NAV Data Points:   {len(nav_values):>14d}")

    assert initial_nav > 0, "Initial NAV must be positive"
    assert not np.isnan(final_nav), "Final NAV is NaN"
    assert len(nav_values) > 5, f"Too few NAV points: {len(nav_values)}"

    # Trade summary
    typer.echo(f"\n### Trade Summary (total: {len(trades)})")
    if trades:
        trade_records = []
        winning = 0
        losing = 0
        total_pnl = 0.0
        pnl_list = []

        for trade in trades:
            asset = getattr(trade, "asset", "?")
            direction = getattr(trade, "direction", "?")
            if hasattr(direction, "value"):
                direction = direction.value
            entry_time = getattr(trade, "entry_time", None)
            entry_price = getattr(trade, "entry_price", 0)
            exit_price = getattr(trade, "exit_price", 0)
            pnl = getattr(trade, "pnl", 0) or 0
            pnl_pct = (getattr(trade, "pnl_pct", 0) or 0) * 100

            if pnl > 0:
                winning += 1
            elif pnl < 0:
                losing += 1
            total_pnl += pnl
            pnl_list.append(pnl)

            trade_records.append({
                "asset": str(asset),
                "direction": str(direction),
                "entry_time": str(entry_time)[:19] if entry_time else "",
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price or 0, 6),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
            })

        # Print first 20 trades
        typer.echo(f"\n  {'#':>4} | {'Asset':>10} | {'Dir':>6} | {'Entry':>20} | {'Entry $':>12} | {'Exit $':>12} | {'PnL':>12} | {'PnL%':>8}")
        typer.echo("  " + "-" * 100)
        for idx, record in enumerate(trade_records[:20]):
            typer.echo(
                f"  {idx+1:>4} | {record['asset']:>10} | {record['direction']:>6} | "
                f"{record['entry_time']:>20} | {record['entry_price']:>12,.4f} | "
                f"{record['exit_price']:>12,.4f} | {record['pnl']:>12,.2f} | "
                f"{record['pnl_pct']:>7.2f}%"
            )
        if len(trade_records) > 20:
            typer.echo(f"  ... and {len(trade_records) - 20} more trades")

        win_rate = winning / len(trades) if trades else 0
        winning_pnls = [p for p in pnl_list if p > 0]
        losing_pnls = [p for p in pnl_list if p < 0]
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls else float("inf")

        typer.echo(f"\n  --- Trade Statistics ---")
        typer.echo(f"  Winning:       {winning:>10d}")
        typer.echo(f"  Losing:        {losing:>10d}")
        typer.echo(f"  Win Rate:      {win_rate * 100:>9.1f}%")
        typer.echo(f"  Total PnL:     {total_pnl:>10,.2f}")
        typer.echo(f"  Avg Win:       {avg_win:>10,.2f}")
        typer.echo(f"  Avg Loss:      {avg_loss:>10,.2f}")
        typer.echo(f"  Profit Factor: {profit_factor:>10.2f}")

        trade_log_path = output_path / "trade_log.json"
        with open(trade_log_path, "w", encoding="utf-8") as file_handle:
            json.dump(trade_records, file_handle, indent=2, ensure_ascii=False)
        logger.info(f"Trade log saved to {trade_log_path}")

    # Save NAV to CSV
    nav_df = pd.DataFrame({"timestamp": nav_timestamps, "nav": nav_values})
    nav_csv_path = output_path / "nav_history.csv"
    nav_df.to_csv(nav_csv_path, index=False)
    logger.info(f"NAV history saved to {nav_csv_path}")

    # Save summary JSON
    summary = {
        "initial_nav": initial_nav,
        "final_nav": final_nav,
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "annualized_vol_pct": round(annualized_vol * 100, 4),
        "annualized_return_pct": round(annualized_return * 100, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "total_trades": len(trades),
        "nav_data_points": len(nav_values),
    }
    summary_path = output_path / "backtest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    typer.echo("\n" + "=" * 70)
    typer.echo("ALL VALIDATIONS PASSED")
    typer.echo("=" * 70)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@pipeline_app.command(help="Run full pipeline: asset selection -> aggregation -> features -> backtest -> validation")
def run(
    data_dir: str = typer.Option("E:/data/binance_futures", help="Tick data directory"),
    bar_dir: str = typer.Option("E:/data/bar_data", help="Output directory for bar data"),
    output_dir: str = typer.Option("E:/data/backtest_results", help="Output directory for results"),
    top_n: int = typer.Option(100, help="Top N assets to select (static pool pre-filter)"),
    threshold: str = typer.Option("1h", help="Bar threshold (e.g., 1h, 5min)"),
    workers: int = typer.Option(4, help="Number of parallel workers for aggregation"),
    capital: float = typer.Option(1_000_000, help="Initial capital"),
    long_n: int = typer.Option(10, help="Number of long positions"),
    short_n: int = typer.Option(10, help="Number of short positions"),
    rebalance: str = typer.Option("W-MON", help="Rebalance frequency"),
    factor: str = typer.Option("return_20", help="Ranking factor column"),
    skip_aggregation: bool = typer.Option(False, help="Skip aggregation (use existing bars)"),
    # --- Dynamic asset pool options ---
    dynamic_pool: bool = typer.Option(
        False,
        help="Enable dynamic asset pool: re-select top assets by dollar-volume at each period",
    ),
    pool_top_n: int = typer.Option(
        100,
        help="(dynamic pool) Number of assets to keep per period",
    ),
    pool_reselect: str = typer.Option(
        "MS",
        help="(dynamic pool) Re-selection frequency, e.g. MS=monthly, QS=quarterly",
    ),
    pool_lookback: str = typer.Option(
        "30D",
        help="(dynamic pool) Lookback window for average dollar-volume, e.g. 30D, 90D",
    ),
):
    """
    End-to-end backtest pipeline.

    When ``--dynamic-pool`` is enabled, *all* available assets (that have
    tick data) are aggregated and featurised, then the backtest engine
    dynamically selects the top assets by rolling dollar-volume at every
    ``--pool-reselect`` interval (e.g. monthly or quarterly).

    Examples:
        pipeline run
        pipeline run --top-n 30 --threshold 1h --workers 8
        pipeline run --skip-aggregation --capital 500000
        pipeline run --dynamic-pool --pool-top-n 50 --pool-reselect MS --pool-lookback 30D
    """
    pipeline_start = time.time()

    typer.echo("=" * 70)
    typer.echo("FULL PIPELINE: SELECT -> AGGREGATE -> FEATURES -> BACKTEST")
    typer.echo("=" * 70)
    typer.echo(f"  Data dir:      {data_dir}")
    typer.echo(f"  Bar dir:       {bar_dir}")
    typer.echo(f"  Output dir:    {output_dir}")
    typer.echo(f"  Top N:         {top_n}")
    typer.echo(f"  Workers:       {workers}")
    typer.echo(f"  Threshold:     {threshold}")
    if dynamic_pool:
        typer.echo(f"  Dynamic pool:  ON (top {pool_top_n}, reselect={pool_reselect}, lookback={pool_lookback})")
    else:
        typer.echo(f"  Dynamic pool:  OFF (static asset pool)")
    typer.echo()

    # Step 1: Determine candidate symbols
    if dynamic_pool:
        # Use ALL available symbols that have tick data on disk
        typer.echo("[Step 1] Collecting all available symbols from tick data directory...")
        step_start = time.time()
        tick_dir = Path(data_dir)
        symbols = sorted([
            d.name for d in tick_dir.iterdir()
            if d.is_dir() and any(d.rglob("*.parquet"))
        ])
        typer.echo(f"  Found {len(symbols)} symbols with tick data ({time.time() - step_start:.1f}s)")
    else:
        typer.echo("[Step 1] Selecting asset pool (static)...")
        step_start = time.time()
        symbols = step_select_asset_pool(top_n, cache_dir=str(Path(data_dir).parent))
        typer.echo(f"  Selected {len(symbols)} assets ({time.time() - step_start:.1f}s)")

    # Step 2: Bar aggregation
    if skip_aggregation:
        typer.echo("\n[Step 2] Skipping aggregation, loading existing bars...")
        bar_paths = {}
        # In dynamic-pool mode, scan the bar directory for ALL existing bars
        scan_symbols = symbols if not dynamic_pool else sorted([
            d.name for d in Path(bar_dir).iterdir()
            if d.is_dir() and any(d.glob("*.parquet"))
        ]) if Path(bar_dir).exists() else symbols
        for symbol in scan_symbols:
            symbol_dir = Path(bar_dir) / symbol
            if symbol_dir.exists():
                parquet_files = sorted(symbol_dir.glob("*.parquet"))
                if parquet_files:
                    bar_paths[symbol] = str(parquet_files[0])
        typer.echo(f"  Found existing bars for {len(bar_paths)} assets")
    else:
        typer.echo(f"\n[Step 2] Aggregating tick data to {threshold} bars...")
        step_start = time.time()
        bar_paths = step_aggregate_bars(
            symbols=symbols,
            tick_data_dir=data_dir,
            bar_output_dir=bar_dir,
            bar_threshold=threshold,
            include_advanced=True,
            max_workers=workers,
        )
        typer.echo(f"  Aggregated {len(bar_paths)} assets ({time.time() - step_start:.1f}s)")
        gc.collect()

    if not bar_paths:
        typer.echo("\n[FAIL] No bar data available. Ensure tick data is downloaded first.")
        raise typer.Exit(code=1)

    # Step 3: Feature calculation
    typer.echo(f"\n[Step 3] Calculating features for {len(bar_paths)} assets...")
    step_start = time.time()
    featured_data = step_calculate_features(bar_paths, feature_windows=[5, 10, 20, 60])
    typer.echo(f"  Features: {featured_data.shape} ({time.time() - step_start:.1f}s)")
    gc.collect()

    # Step 4: Backtest
    typer.echo("\n[Step 4] Running cross-sectional backtest...")
    step_start = time.time()
    backtest_output = step_run_backtest(
        data=featured_data,
        initial_capital=capital,
        top_n_long=long_n,
        top_n_short=short_n,
        rebalance_frequency=rebalance,
        ranking_factor=factor,
        dynamic_pool=dynamic_pool,
        pool_top_n=pool_top_n,
        pool_reselect_frequency=pool_reselect,
        pool_lookback_period=pool_lookback,
    )
    typer.echo(f"  Backtest completed ({time.time() - step_start:.1f}s)")

    # Step 5: Validate and report
    typer.echo("\n[Step 5] Validating results and saving reports...")
    step_validate_and_report(backtest_output, output_dir)

    total_elapsed = time.time() - pipeline_start
    typer.echo(f"\nTotal pipeline time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
