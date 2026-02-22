"""
Data CLI commands: download, list, info, convert, funding-rate.
"""
import concurrent.futures
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from tqdm import tqdm

data_app = typer.Typer(help="Tick data management (download / list / inspect / convert)")


@data_app.command(help="Download tick data from exchange")
def download(
    exchange: str = typer.Option("binance_futures", help="Exchange name (binance, binance_futures, okx_futures)"),
    symbols: Optional[List[str]] = typer.Option(None, help="Symbols to download (default: all)"),
    start_date: str = typer.Option("auto", help="Start date in YYYY-MM format, or 'auto' for earliest pair data"),
    end_date: str = typer.Option("auto", help="End date in YYYY-MM format, or 'auto' for current month"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Override output directory (e.g. E:/data/okx_futures)",
    ),
    threads: int = typer.Option(8, help="Number of concurrent download threads"),
):
    """
    Download tick data and convert to Parquet.

    Examples:
        data download --start-date 2025-01 --end-date 2025-06
        data download --exchange binance --symbols BTCUSDT ETHUSDT --start-date 2025-01 --end-date 2025-03
        data download --exchange okx_futures --start-date auto --end-date auto
    """
    from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download

    typer.echo(f"[*] Downloading {exchange} tick data ({start_date} -> {end_date})")
    if symbols:
        typer.echo(f"[*] Symbols: {', '.join(symbols)}")
    else:
        typer.echo("[*] Symbols: all available")
    if data_dir:
        typer.echo(f"[*] Output dir: {data_dir}")
    typer.echo(f"[*] Threads: {threads}")

    try:
        run_download(
            exchange_name=exchange,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
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


@data_app.command(help="Extract ZIP files and convert to Parquet")
def convert(
    data_dir: str = typer.Option("E:/data/binance_futures", help="Root directory containing symbol sub-folders with ZIP files"),
    symbol: Optional[str] = typer.Option(None, help="Only convert a specific symbol (e.g. BTCUSDT)"),
    workers: int = typer.Option(4, help="Number of parallel conversion processes"),
    force: bool = typer.Option(False, help="Re-convert even if Parquet already exists"),
):
    """
    Scan data directory for ZIP files, extract and convert to Parquet.

    Useful when ZIP downloads completed but Parquet conversion was skipped or interrupted.

    Examples:
        data convert --data-dir E:/data/binance_futures
        data convert --symbol BTCUSDT --force
        data convert --workers 8
    """
    root_path = Path(data_dir)
    if not root_path.exists():
        typer.echo(f"[!] Data directory not found: {data_dir}")
        raise typer.Exit(code=1)

    # Collect symbol directories
    if symbol:
        symbol_dirs = [root_path / symbol.upper()]
        if not symbol_dirs[0].exists():
            typer.echo(f"[!] Symbol directory not found: {symbol_dirs[0]}")
            raise typer.Exit(code=1)
    else:
        symbol_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    # Scan for ZIP files that need conversion
    pending_zips: List[Path] = []
    skipped_count = 0

    for symbol_dir in symbol_dirs:
        for zip_file in sorted(symbol_dir.glob("*.zip")):
            # Parquet should be at the same level as ZIP: {symbol_dir}/{stem}.parquet
            expected_parquet = symbol_dir / f"{zip_file.stem}.parquet"

            if expected_parquet.exists() and not force:
                skipped_count += 1
            else:
                pending_zips.append(zip_file)

    typer.echo(f"[*] Scanned {len(symbol_dirs)} symbol directories")
    typer.echo(f"[*] Found {len(pending_zips)} ZIP files to convert, {skipped_count} already have Parquet (skipped)")

    if not pending_zips:
        typer.echo("[+] Nothing to convert")
        return

    # Reuse the module-level function from downloader (Windows-safe for ProcessPoolExecutor)
    from crypto_data_engine.services.tick_data_scraper.downloader.downloader import (
        _process_single_zip,
    )

    zip_path_strings = [str(zp) for zp in pending_zips]
    successful = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_single_zip, zip_str): zip_str
            for zip_str in zip_path_strings
        }
        with tqdm(total=len(zip_path_strings), desc="[Extract & Convert]") as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                zip_str = futures[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                        typer.echo(f"  [!] Failed: {Path(zip_str).name}")
                except Exception as error:
                    failed += 1
                    typer.echo(f"  [!] Error: {Path(zip_str).name}: {error}")
                progress_bar.update(1)

    typer.echo(f"[+] Convert completed: {successful} succeeded, {failed} failed")


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


@data_app.command(name="funding-rate", help="Download Binance Futures funding rate history")
def funding_rate(
    symbols: Optional[List[str]] = typer.Option(
        None, help="Symbols to download (default: all perpetual USDT)"
    ),
    start_date: str = typer.Option(..., help="Start date in YYYY-MM format"),
    end_date: str = typer.Option(..., help="End date in YYYY-MM format"),
    threads: int = typer.Option(4, help="Number of concurrent download threads"),
    output_dir: str = typer.Option(
        "E:/data/funding_rate",
        help="Output directory for funding rate parquet files",
    ),
):
    """
    Download Binance Futures funding rate data via REST API.

    Data is paginated from /fapi/v1/fundingRate and stored as monthly Parquet
    files. Supports incremental downloads (skips existing months).

    Examples:
        data funding-rate --start-date 2020-01 --end-date 2025-12
        data funding-rate --symbols BTCUSDT ETHUSDT --start-date 2024-01 --end-date 2024-12
    """
    from crypto_data_engine.services.funding_rate.downloader import (
        FundingRateDownloader,
    )

    out_path = Path(output_dir)
    downloader = FundingRateDownloader(output_dir=out_path)

    if symbols:
        sym_list = [s.upper() for s in symbols]
    else:
        typer.echo("[*] Fetching all perpetual USDT-M futures symbols...")
        sym_list = downloader.get_symbols()

    typer.echo(f"[*] Downloading funding rates for {len(sym_list)} symbols")
    typer.echo(f"[*] Date range: {start_date} -> {end_date}")
    typer.echo(f"[*] Output: {out_path}")
    typer.echo(f"[*] Threads: {threads}")

    total_new = 0
    total_skipped = 0
    errors = 0

    def _download_one(sym: str):
        try:
            count = downloader.download_symbol_range(sym, start_date, end_date)
            return sym, count, None
        except Exception as e:
            return sym, 0, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(_download_one, s): s for s in sym_list}
        with tqdm(total=len(sym_list), desc="[Funding Rate]") as pbar:
            for future in concurrent.futures.as_completed(futures):
                sym, count, err = future.result()
                if err:
                    errors += 1
                    typer.echo(f"  [!] {sym}: {err}")
                elif count > 0:
                    total_new += count
                else:
                    total_skipped += 1
                pbar.update(1)

    typer.echo(
        f"[+] Funding rate download complete: "
        f"{total_new} new months, {total_skipped} symbols up-to-date, "
        f"{errors} errors"
    )
