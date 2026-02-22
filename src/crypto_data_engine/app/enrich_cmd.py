"""
CLI commands for enriching dollar bars with tick microstructure features.

Usage:
    main enrich single BTCUSDT
    main enrich batch --workers 10
    main enrich batch --workers 10 --symbols BTCUSDT ETHUSDT
"""
from pathlib import Path
from typing import List, Optional

import typer

enrich_app = typer.Typer(
    help="Enrich bars with tick microstructure features",
    no_args_is_help=True,
)

DEFAULT_BAR_DIR = "E:/data/dollar_bar/bars"
DEFAULT_TICK_DIR = "E:/data/binance_futures"
DEFAULT_OUTPUT_DIR = "E:/data/dollar_bar/bars_enriched"


def _find_file_pairs(
    symbol: str,
    bar_dir: Path,
    tick_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> list:
    """Find matching (bar, tick, output) file triples for a symbol."""
    sym_bar_dir = bar_dir / symbol
    sym_tick_dir = tick_dir / symbol
    sym_out_dir = output_dir / symbol

    if not sym_bar_dir.exists() or not sym_tick_dir.exists():
        return []

    bar_files = {f.stem: f for f in sym_bar_dir.glob("*.parquet")}
    # Map bar filename to matching tick file via year-month
    # Bar: BTCUSDT_dollar_bar_auto_K50_ema_2024-01.parquet → 2024-01
    # Tick: BTCUSDT-aggTrades-2024-01.parquet → 2024-01
    pairs = []
    for bar_stem, bar_path in sorted(bar_files.items()):
        parts = bar_stem.rsplit("_", 1)
        if len(parts) < 2:
            continue
        year_month = parts[-1]  # e.g. "2024-01"

        tick_file = sym_tick_dir / f"{symbol}-aggTrades-{year_month}.parquet"
        if not tick_file.exists():
            continue

        out_file = sym_out_dir / bar_path.name
        if out_file.exists() and not force:
            continue

        pairs.append((str(bar_path), str(tick_file), str(out_file)))

    return pairs


@enrich_app.command(name="single", help="Enrich bars for a single symbol")
def enrich_single(
    symbol: str = typer.Argument(..., help="Symbol (e.g. BTCUSDT)"),
    bar_dir: str = typer.Option(DEFAULT_BAR_DIR, help="Bar data directory"),
    tick_dir: str = typer.Option(DEFAULT_TICK_DIR, help="Tick data directory"),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory"),
    lookback_bars: int = typer.Option(5, help="Rolling window size in bars"),
    force: bool = typer.Option(False, help="Overwrite existing enriched files"),
):
    """Enrich all bar files for a single symbol with tick features."""
    from crypto_data_engine.services.bar_aggregator.tick_feature_enricher import (
        TickFeatureEnricher,
        TickFeatureEnricherConfig,
    )

    bar_p, tick_p, out_p = Path(bar_dir), Path(tick_dir), Path(output_dir)
    symbol = symbol.upper()

    pairs = _find_file_pairs(symbol, bar_p, tick_p, out_p, force)
    if not pairs:
        typer.echo(f"[!] No file pairs found for {symbol} (all done or missing data)")
        return

    typer.echo(f"[*] Enriching {symbol}: {len(pairs)} files")
    out_sym_dir = out_p / symbol
    out_sym_dir.mkdir(parents=True, exist_ok=True)

    config = TickFeatureEnricherConfig(lookback_bars=lookback_bars)
    enricher = TickFeatureEnricher(config)

    for bar_path, tick_path, output_path in pairs:
        result = enricher.enrich_file_pair(bar_path, tick_path, output_path)
        status = result["status"]
        tag = "✓" if status == "ok" else "✗"
        typer.echo(
            f"  {tag} {Path(bar_path).name}: "
            f"{result['n_bars']} bars, {result['n_ticks']:,} ticks, "
            f"{result['n_enriched']} enriched — {status}"
        )

    typer.echo(f"[+] Done: {symbol}")


@enrich_app.command(name="batch", help="Batch enrich bars with tick features")
def enrich_batch(
    bar_dir: str = typer.Option(DEFAULT_BAR_DIR, help="Bar data directory"),
    tick_dir: str = typer.Option(DEFAULT_TICK_DIR, help="Tick data directory"),
    output_dir: str = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory"),
    symbols: Optional[List[str]] = typer.Option(None, help="Specific symbols"),
    workers: int = typer.Option(10, help="Parallel workers"),
    lookback_bars: int = typer.Option(5, help="Rolling window size in bars"),
    force: bool = typer.Option(False, help="Overwrite existing enriched files"),
    top_n: Optional[int] = typer.Option(None, help="Only process top N symbols by file count"),
):
    """
    Batch enrich all symbols with tick microstructure features.

    Scans bar and tick directories, finds matching file pairs,
    and processes them in parallel.

    Examples:
        enrich batch --workers 10
        enrich batch --symbols BTCUSDT ETHUSDT --workers 4
        enrich batch --top-n 50 --workers 10
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from crypto_data_engine.services.bar_aggregator.tick_feature_enricher import (
        enrich_file_pair_worker,
    )

    bar_p, tick_p, out_p = Path(bar_dir), Path(tick_dir), Path(output_dir)

    # Discover symbols
    if symbols:
        sym_list = [s.upper() for s in symbols]
    else:
        sym_list = sorted(
            d.name
            for d in bar_p.iterdir()
            if d.is_dir() and (tick_p / d.name).exists()
        )

    typer.echo(f"[*] Found {len(sym_list)} symbols with both bar and tick data")

    # Collect all file pairs
    all_pairs = []
    for sym in sym_list:
        pairs = _find_file_pairs(sym, bar_p, tick_p, out_p, force)
        all_pairs.extend(pairs)

    if top_n and len(sym_list) > top_n:
        # Sort by number of pairs descending, take top N
        sym_counts = {}
        for bp, tp, op in all_pairs:
            sym = Path(bp).parent.name
            sym_counts[sym] = sym_counts.get(sym, 0) + 1
        top_syms = set(
            s for s, _ in sorted(sym_counts.items(), key=lambda x: -x[1])[:top_n]
        )
        all_pairs = [(b, t, o) for b, t, o in all_pairs if Path(b).parent.name in top_syms]
        typer.echo(f"[*] Filtered to top {top_n} symbols: {len(all_pairs)} files")

    if not all_pairs:
        typer.echo("[!] No files to process (all done or no matching pairs)")
        return

    # Ensure output directories
    out_syms = set(Path(o).parent for _, _, o in all_pairs)
    for d in out_syms:
        Path(d).mkdir(parents=True, exist_ok=True)

    typer.echo(f"[*] Processing {len(all_pairs)} files with {workers} workers")

    # Build worker args
    cfg_dict = dict(lookback_bars=lookback_bars)
    worker_args = [
        (bp, tp, op, cfg_dict) for bp, tp, op in all_pairs
    ]

    t0 = time.time()
    done = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(enrich_file_pair_worker, args): args[0]
            for args in worker_args
        }
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                if result["status"] != "ok":
                    errors += 1
                    typer.echo(f"  ✗ {Path(result['bar_path']).name}: {result['status']}")
                elif done % 50 == 0 or done == len(all_pairs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(all_pairs) - done) / rate if rate > 0 else 0
                    typer.echo(
                        f"  [{done}/{len(all_pairs)}] "
                        f"{elapsed:.0f}s elapsed, {rate:.1f} files/s, "
                        f"ETA {eta:.0f}s"
                    )
            except Exception as e:
                errors += 1
                typer.echo(f"  ✗ {Path(futures[future]).name}: {e}")

    elapsed = time.time() - t0
    typer.echo(
        f"\n[+] Batch enrich complete: {done} files in {elapsed:.0f}s "
        f"({errors} errors)"
    )
    typer.echo(f"[+] Output: {out_p}")
