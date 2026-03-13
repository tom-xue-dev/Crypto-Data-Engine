"""
Data CLI commands: list symbols, inspect data, load preview, download.
"""
from pathlib import Path
from typing import List, Optional

import typer

data_app = typer.Typer(help="Data loading, inspection and download", no_args_is_help=True)


@data_app.command(name="list", help="List available symbols")
def list_symbols(
    bar_type: str = typer.Option("time", help="Bar type"),
    interval: str = typer.Option("1h", help="Bar interval"),
    source: str = typer.Option("bar", help="Data source: bar or tick"),
):
    """
    List all available symbols.

    Examples:
        data list
        data list --source tick
        data list --interval 5m
    """
    from crypto_data_engine.services.data_manager import BarDataLoader, TickDataLoader

    if source == "bookticker":
        from crypto_data_engine.services.data_manager import BookTickerDataLoader
        loader = BookTickerDataLoader()
        symbols = loader.list_symbols()
    elif source == "tick":
        loader = TickDataLoader()
        symbols = loader.list_symbols()
    else:
        loader = BarDataLoader()
        symbols = loader.list_symbols(bar_type=bar_type, interval=interval)

    if not symbols:
        typer.echo("[!] No symbols found")
        raise typer.Exit(code=1)

    typer.echo(f"Available symbols ({len(symbols)}):\n")
    # Print in columns
    cols = 5
    for i in range(0, len(symbols), cols):
        row = symbols[i : i + cols]
        typer.echo("  ".join(f"{s:<16}" for s in row))


@data_app.command(name="info", help="Show date range and shape for a symbol")
def info(
    symbol: str = typer.Argument(..., help="Symbol name (e.g. BTCUSDT)"),
    bar_type: str = typer.Option("time", help="Bar type"),
    interval: str = typer.Option("1h", help="Bar interval"),
):
    """
    Show data info for a symbol.

    Examples:
        data info BTCUSDT
        data info ETHUSDT --interval 5m
    """
    from crypto_data_engine.services.data_manager import BarDataLoader

    loader = BarDataLoader()
    try:
        first, last = loader.get_date_range(symbol, bar_type=bar_type, interval=interval)
    except FileNotFoundError as exc:
        typer.echo(f"[!] {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"Symbol:    {symbol}")
    typer.echo(f"Bar type:  {bar_type}/{interval}")
    typer.echo(f"Range:     {first} -> {last}")

    # Load one month to show columns
    data = loader.load([symbol], first, first, bar_type=bar_type, interval=interval)
    if data and symbol in data:
        df = data[symbol]
        typer.echo(f"Rows:      {len(df)} (first month)")
        typer.echo(f"Columns:   {len(df.columns)}")
        typer.echo(f"\nColumn list:")
        cols = list(df.columns)
        per_row = 4
        for i in range(0, len(cols), per_row):
            row = cols[i : i + per_row]
            typer.echo("  " + ", ".join(row))


@data_app.command(name="head", help="Preview first N rows of bar data")
def head(
    symbol: Optional[str] = typer.Argument(None, help="Symbol name (omit for panel view)"),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM)"),
    end_date: Optional[str] = typer.Option(None, "--end", help="End date (default: same as start)"),
    rows: int = typer.Option(10, "-n", help="Number of rows to show"),
    columns: Optional[List[str]] = typer.Option(
        None, "-c", help="Columns to show (default: OHLCV)"
    ),
    bar_type: str = typer.Option("time", help="Bar type"),
    interval: str = typer.Option("1h", help="Bar interval"),
):
    """
    Preview bar data. Omit symbol to show panel (all symbols, MultiIndex).

    Examples:
        data head BTCUSDT --start 2024-06
        data head BTCUSDT --start 2024-06 -n 20 -c close -c volume -c amihud
        data head --start 2024-06 -n 20
    """
    from crypto_data_engine.services.data_manager import BarDataLoader

    loader = BarDataLoader()
    end = end_date or start_date
    col_list = list(columns) if columns else ["open", "high", "low", "close", "volume"]

    if symbol:
        data = loader.load(
            [symbol], start_date, end,
            bar_type=bar_type, interval=interval, columns=col_list,
        )
        if not data or symbol not in data:
            typer.echo(f"[!] No data for {symbol} in {start_date} -> {end}")
            raise typer.Exit(code=1)
        df = data[symbol]
        typer.echo(f"{symbol} ({start_date} -> {end}): {len(df)} rows total\n")
        typer.echo(df.head(rows).to_string())
    else:
        panel = loader.load_panel(
            None, start_date, end,
            bar_type=bar_type, interval=interval, columns=col_list,
        )
        if panel.empty:
            typer.echo(f"[!] No data in {start_date} -> {end}")
            raise typer.Exit(code=1)
        n_symbols = panel.index.get_level_values("symbol").nunique()
        typer.echo(f"Panel ({start_date} -> {end}): {len(panel)} rows, {n_symbols} symbols\n")
        typer.echo(panel.head(rows).to_string())


@data_app.command(name="tick-head", help="Preview tick data")
def tick_head(
    symbol: str = typer.Argument(..., help="Symbol name"),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM)"),
    rows: int = typer.Option(10, "-n", help="Number of rows"),
):
    """
    Preview raw tick (aggTrades) data.

    Examples:
        data tick-head BTCUSDT --start 2024-06
    """
    from crypto_data_engine.services.data_manager import TickDataLoader

    loader = TickDataLoader()
    df = loader.load(symbol, start_date, start_date)

    if df.empty:
        typer.echo(f"[!] No tick data for {symbol} in {start_date}")
        raise typer.Exit(code=1)

    typer.echo(f"{symbol} ticks ({start_date}): {len(df):,} rows total\n")
    typer.echo(df.head(rows).to_string())


@data_app.command(name="bookticker-head", help="Preview bookTicker data")
def bookticker_head(
    symbol: str = typer.Argument(..., help="Symbol name"),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM)"),
    rows: int = typer.Option(10, "-n", help="Number of rows"),
):
    """
    Preview raw bookTicker data.

    Examples:
        data bookticker-head BTCUSDT --start 2023-06
    """
    from crypto_data_engine.services.data_manager import BookTickerDataLoader

    loader = BookTickerDataLoader()
    df = loader.load(symbol, start_date, start_date)

    if df.empty:
        typer.echo(f"[!] No bookTicker data for {symbol} in {start_date}")
        raise typer.Exit(code=1)

    typer.echo(f"{symbol} bookTicker ({start_date}): {len(df):,} rows total\n")
    typer.echo(df.head(rows).to_string())


@data_app.command(name="download", help="Download data from exchange")
def download(
    exchange: str = typer.Option(
        "binance_futures", help="Exchange name (binance, binance_futures, binance_futures_bookticker, okx_futures)"
    ),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM or 'auto')"),
    end_date: str = typer.Option(..., "--end", help="End date (YYYY-MM or 'auto')"),
    symbols: Optional[List[str]] = typer.Option(None, "-s", help="Symbols to download (default: all)"),
    data_dir: Optional[str] = typer.Option(None, help="Override output directory"),
    threads: int = typer.Option(8, help="Number of download threads"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Path to YAML config file"),
):
    """
    Download exchange data (aggTrades, bookTicker, etc.).

    Examples:
        data download --exchange binance_futures --start 2024-01 --end 2024-06
        data download --exchange binance_futures_bookticker --start 2023-05 --end 2024-04
        data download --exchange binance_futures_bookticker --start 2023-05 --end 2024-04 -s BTCUSDT -s ETHUSDT
        data download --exchange binance_futures --start auto --end auto --config my_config.yaml
    """
    from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download

    symbol_list = list(symbols) if symbols else None

    typer.echo(f"[*] Starting download: {exchange}")
    typer.echo(f"    Date range: {start_date} -> {end_date}")
    if symbol_list:
        typer.echo(f"    Symbols: {', '.join(symbol_list)}")
    else:
        typer.echo(f"    Symbols: all available")

    result = run_download(
        exchange_name=exchange,
        symbols=symbol_list,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        max_threads=threads,
    )
    typer.echo(f"[+] Download completed: {result}")
