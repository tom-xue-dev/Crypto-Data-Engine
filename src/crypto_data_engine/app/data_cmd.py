"""
Data CLI commands: download, list, info.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

data_app = typer.Typer(help="Tick data management (download / list / inspect)")


@data_app.command(help="Download tick data from exchange")
def download(
    exchange: str = typer.Option("binance_futures", help="Exchange name (binance, binance_futures, okx)"),
    symbols: Optional[List[str]] = typer.Option(None, help="Symbols to download (default: all)"),
    start_date: str = typer.Option(..., help="Start date in YYYY-MM format"),
    end_date: str = typer.Option(..., help="End date in YYYY-MM format"),
    threads: int = typer.Option(8, help="Number of concurrent download threads"),
):
    """
    Download tick data and convert to Parquet.

    Examples:
        data download --start-date 2025-01 --end-date 2025-06
        data download --exchange binance --symbols BTCUSDT ETHUSDT --start-date 2025-01 --end-date 2025-03
    """
    from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download

    typer.echo(f"[*] Downloading {exchange} tick data ({start_date} -> {end_date})")
    if symbols:
        typer.echo(f"[*] Symbols: {', '.join(symbols)}")
    else:
        typer.echo("[*] Symbols: all available")
    typer.echo(f"[*] Threads: {threads}")

    try:
        run_download(
            exchange_name=exchange,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            max_threads=threads,
        )
        typer.echo("[+] Download completed")
    except Exception as error:
        typer.echo(f"[!] Download failed: {error}")
        raise typer.Exit(code=1)


@data_app.command(name="list", help="List available tick data")
def list_data(
    data_dir: str = typer.Option("./data/tick_data", help="Tick data directory"),
    symbol: Optional[str] = typer.Option(None, help="Filter by symbol"),
):
    """List available tick data files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        typer.echo(f"[!] Data directory not found: {data_dir}")
        return

    symbols = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

    if symbol:
        symbols = [s for s in symbols if symbol.upper() in s]

    typer.echo(f"[*] Found {len(symbols)} symbols in {data_dir}")

    for symbol_name in symbols[:20]:
        symbol_dir = data_path / symbol_name
        files = list(symbol_dir.glob("*.parquet"))
        typer.echo(f"  {symbol_name}: {len(files)} files")

    if len(symbols) > 20:
        typer.echo(f"  ... and {len(symbols) - 20} more symbols")


@data_app.command(help="Show data info for a symbol")
def info(
    symbol: str = typer.Argument(..., help="Symbol to inspect (e.g., BTCUSDT)"),
    data_dir: str = typer.Option("./data/tick_data", help="Tick data directory"),
):
    """Show detailed info about tick data for a symbol."""
    data_path = Path(data_dir) / symbol.upper()
    if not data_path.exists():
        typer.echo(f"[!] Symbol directory not found: {data_path}")
        return

    files = sorted(data_path.glob("*.parquet"))
    typer.echo(f"\n[*] {symbol.upper()} tick data:")
    typer.echo(f"  Directory: {data_path}")
    typer.echo(f"  Files: {len(files)}")

    if files:
        dates = []
        for file in files:
            parts = file.stem.split("-")
            if len(parts) >= 4:
                dates.append(f"{parts[-2]}-{parts[-1]}")

        if dates:
            typer.echo(f"  Date range: {min(dates)} to {max(dates)}")

        try:
            sample_df = pd.read_parquet(files[0])
            typer.echo(f"  Rows per file: ~{len(sample_df):,}")
            typer.echo(f"  Total estimated rows: ~{len(sample_df) * len(files):,}")
        except Exception as error:
            typer.echo(f"  [!] Error reading file: {error}")
