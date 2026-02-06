"""
Bar aggregation CLI command.
"""
from pathlib import Path

import pandas as pd
import typer

aggregate_app = typer.Typer(help="Aggregate tick data to bars")


@aggregate_app.callback(invoke_without_command=True)
def aggregate(
    symbol: str = typer.Argument(..., help="Symbol to aggregate (e.g., BTCUSDT)"),
    bar_type: str = typer.Option("dollar_bar", help="Bar type: time_bar, tick_bar, volume_bar, dollar_bar"),
    threshold: str = typer.Option("1000000", help="Threshold (e.g., 1000000 for dollar, 1000 for tick, 5min for time)"),
    data_dir: str = typer.Option("./data/tick_data", help="Tick data directory"),
    output_dir: str = typer.Option("./data/bars", help="Output directory for bars"),
    use_numba: bool = typer.Option(True, help="Use Numba acceleration if available"),
):
    """
    Aggregate tick data to bars using the unified bar aggregator.

    Examples:
        aggregate BTCUSDT --bar-type dollar_bar --threshold 1000000
        aggregate BTCUSDT --bar-type time_bar --threshold 5min
        aggregate BTCUSDT --bar-type tick_bar --threshold 1000
    """
    from crypto_data_engine.services.bar_aggregator import (
        aggregate_bars,
        NUMBA_AVAILABLE,
    )

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
    typer.echo("[*] Loading tick data...")

    dfs = []
    for file in files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as error:
            typer.echo(f"[!] Error loading {file.name}: {error}")

    if not dfs:
        typer.echo("[!] No data loaded")
        raise typer.Exit(code=1)

    tick_data = pd.concat(dfs, ignore_index=True)
    typer.echo(f"[+] Loaded {len(tick_data):,} ticks")

    # Parse threshold
    try:
        if bar_type in ["dollar_bar", "volume_bar"]:
            threshold_val = float(threshold)
        elif bar_type == "tick_bar":
            threshold_val = int(threshold)
        else:
            threshold_val = threshold
    except ValueError:
        threshold_val = threshold

    typer.echo(f"[*] Aggregating... (Numba: {'enabled' if use_numba and NUMBA_AVAILABLE else 'disabled'})")

    try:
        bars = aggregate_bars(
            tick_data,
            bar_type,
            threshold_val,
            use_numba=use_numba,
        )

        typer.echo(f"[+] Generated {len(bars)} bars")

        output_path = Path(output_dir) / symbol.upper()
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"{symbol}_{bar_type}_{threshold}.parquet"
        bars.to_parquet(output_file)

        typer.echo(f"[+] Saved to {output_file}")

        if len(bars) > 0:
            typer.echo("\nBar Summary:")
            typer.echo(f"  Time range: {bars['start_time'].iloc[0]} to {bars['start_time'].iloc[-1]}")
            typer.echo(f"  Avg volume: {bars['volume'].mean():,.2f}")
            if "dollar_volume" in bars.columns:
                typer.echo(f"  Avg dollar volume: ${bars['dollar_volume'].mean():,.2f}")

    except Exception as error:
        typer.echo(f"[!] Aggregation failed: {error}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
