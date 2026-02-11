"""
Bar aggregation CLI commands.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

aggregate_app = typer.Typer(help="Aggregate tick data to bars", no_args_is_help=True)


@aggregate_app.command(name="single", help="Aggregate a single symbol to bars")
def aggregate(
    symbol: str = typer.Argument(..., help="Symbol to aggregate (e.g., BTCUSDT)"),
    bar_type: str = typer.Option("dollar_bar", help="Bar type: time_bar, tick_bar, volume_bar, dollar_bar"),
    threshold: str = typer.Option("1000000", help="Threshold (e.g., 1000000 for dollar, 1000 for tick, 5min for time)"),
    data_dir: str = typer.Option("E:/data/binance_futures", help="Tick data directory"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory (default: E:/data/{bar_type}/bars)"),
    use_numba: bool = typer.Option(True, help="Use Numba acceleration if available"),
):
    """
    Aggregate tick data to bars using the unified bar aggregator.

    Examples:
        aggregate BTCUSDT --bar-type dollar_bar --threshold 1000000
        aggregate BTCUSDT --bar-type time_bar --threshold 5min
        aggregate BTCUSDT --bar-type tick_bar --threshold 1000
    """
    from crypto_data_engine.services.bar_aggregator import aggregate_bars, NUMBA_AVAILABLE
    from crypto_data_engine.services.bar_aggregator.tick_normalizer import normalize_tick_data

    # Default output directory: E:/data/{bar_type}/bars
    if output_dir is None:
        output_dir = f"E:/data/{bar_type}/bars"

    typer.echo(f"[*] Aggregating {symbol} to {bar_type} (threshold: {threshold})")

    tick_path = Path(data_dir) / symbol.upper()
    if not tick_path.exists():
        typer.echo(f"[!] Tick data not found: {tick_path}")
        raise typer.Exit(code=1)

    files = sorted(tick_path.glob("*.parquet"))
    if not files:
        typer.echo(f"[!] No parquet files found in {tick_path}")
        raise typer.Exit(code=1)

    typer.echo(f"[*] Found {len(files)} tick data files")
    typer.echo(f"[*] Numba: {'enabled' if use_numba and NUMBA_AVAILABLE else 'disabled'}")

    # Parse threshold once
    try:
        if bar_type in ["dollar_bar", "volume_bar"]:
            threshold_val = float(threshold)
        elif bar_type == "tick_bar":
            threshold_val = int(threshold)
        else:
            threshold_val = threshold
    except ValueError:
        threshold_val = threshold

    # Process each file individually (memory-efficient, uses fast Numba kernel)
    output_base = Path(output_dir) / symbol.upper()
    output_base.mkdir(parents=True, exist_ok=True)

    total_bars = 0
    total_ticks = 0

    try:
        for file in files:
            try:
                raw = pd.read_parquet(file)
                tick_data = normalize_tick_data(raw, source_hint=file.name)
            except Exception as error:
                typer.echo(f"  [!] Skip {file.name}: {error}")
                continue

            total_ticks += len(tick_data)

            bars = aggregate_bars(
                tick_data, bar_type, threshold_val, use_numba=use_numba,
            )

            if len(bars) > 0:
                # Extract year-month from filename for output naming
                stem = file.stem
                parts = stem.split("-")
                year_month = f"{parts[-2]}-{parts[-1]}" if len(parts) >= 4 else "unknown"
                output_file = output_base / f"{symbol}_{bar_type}_{threshold}_{year_month}.parquet"
                bars.to_parquet(output_file)
                total_bars += len(bars)
                typer.echo(f"  {file.name} -> {len(bars)} bars")

        typer.echo(f"\n[+] Done: {total_ticks:,} ticks -> {total_bars:,} bars ({len(files)} files)")
        typer.echo(f"[+] Output: {output_base}")

    except Exception as error:
        typer.echo(f"[!] Aggregation failed: {error}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@aggregate_app.command(name="batch", help="Batch aggregate all tick data to bars")
def batch_aggregate(
    tick_dir: str = typer.Option("E:/data/binance_futures", help="Tick data root directory"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory (default: E:/data/{bar_type}/bars)"),
    bar_type: str = typer.Option("dollar_bar", help="Bar type (time_bar, tick_bar, volume_bar, dollar_bar)"),
    threshold: str = typer.Option("1000000", help="Threshold (5min for time, 1M for dollar, 'auto' for dynamic dollar bar)"),
    symbols: Optional[List[str]] = typer.Option(None, help="Specific symbols to aggregate (omit for all symbols)"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
    force: bool = typer.Option(False, help="Re-aggregate even if bar file exists"),
    lookback_days: int = typer.Option(10, help="[auto mode] Rolling lookback window in days (N)"),
    bars_per_day: int = typer.Option(50, help="[auto mode] Target number of bars per day (K)"),
    discard_months: int = typer.Option(1, help="[auto mode] Discard first M months of newly listed pairs"),
    use_ema: bool = typer.Option(False, help="[auto mode] Use EMA instead of SMA for rolling average"),
):
    """
    Batch aggregate all tick data to bars using Redis-backed pipeline.

    Automatically scans tick data directory, skips already-aggregated files,
    and processes all pending tasks in parallel.

    By default (no --symbols flag), processes ALL symbols found in tick_dir.
    Output directory defaults to E:/data/{bar_type}/bars.

    For dollar_bar with --threshold auto, uses dynamic thresholds:
      SMA: threshold = mean(daily_dollar_volume, N days) / K
      EMA: threshold = EMA(daily_dollar_volume, span=N).last / K  (--use-ema)

    Examples:
        aggregate batch --bar-type dollar_bar --threshold auto              # Dynamic (SMA)
        aggregate batch --bar-type dollar_bar --threshold auto --use-ema    # Dynamic (EMA)
        aggregate batch --bar-type dollar_bar --threshold auto --bars-per-day 100
        aggregate batch --bar-type time_bar --threshold 5min                # 5-minute time bars
        aggregate batch --bar-type dollar_bar --threshold 1000000           # Fixed 1M dollar bars
        aggregate batch --symbols BTCUSDT ETHUSDT --workers 8
        aggregate batch --force                                             # Re-aggregate all
    """
    from crypto_data_engine.services.bar_aggregator.batch_aggregator import BatchAggregator

    # Default output directory: E:/data/{bar_type}/bars
    if output_dir is None:
        output_dir = f"E:/data/{bar_type}/bars"

    is_auto = threshold.lower() == "auto"
    avg_method = "EMA" if use_ema else "SMA"

    typer.echo(f"[*] Batch aggregation: {bar_type}")
    if is_auto:
        typer.echo(
            f"[*] Threshold: auto ({avg_method}, lookback={lookback_days}d, "
            f"K={bars_per_day} bars/day, discard={discard_months}mo)"
        )
    else:
        typer.echo(f"[*] Threshold: {threshold}")
    typer.echo(f"[*] Tick data: {tick_dir}")
    typer.echo(f"[*] Output: {output_dir}")
    if symbols:
        typer.echo(f"[*] Symbols: {', '.join(symbols)}")
    else:
        typer.echo("[*] Symbols: ALL (scanning entire tick data directory)")
    typer.echo(f"[*] Workers: {workers}")

    try:
        aggregator = BatchAggregator(
            tick_data_dir=tick_dir,
            output_dir=output_dir,
            bar_type=bar_type,
            threshold=threshold,
            lookback_days=lookback_days,
            bars_per_day=bars_per_day,
            discard_months=discard_months,
            use_ema=use_ema,
        )

        aggregator.run_aggregation_pipeline(
            symbols=symbols,
            workers=workers,
            force=force,
        )

        typer.echo("[+] Batch aggregation completed")

    except Exception as error:
        typer.echo(f"[!] Batch aggregation failed: {error}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
