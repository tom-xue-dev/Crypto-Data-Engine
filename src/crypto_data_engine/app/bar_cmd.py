"""
Bar aggregation CLI commands.

Delegates all computation to the C++ bar_aggregator binary located at
{project_root}/bin/bar_aggregator(.exe).

Examples:
    main bar aggregate -s BTCUSDT -s ETHUSDT --start 2024-01 --end 2024-06
    main bar aggregate --bar-type time_bar --threshold 1h --data-type bookticker \
        --start 2024-01 --end 2024-06
    main bar aggregate --bar-type dollar_bar --threshold auto \
        --start 2020-01 --end 2024-12 --threads 4
"""
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import typer

bar_app = typer.Typer(help="Bar aggregation (C++ streaming backend)", no_args_is_help=True)


def _find_binary() -> Path:
    """Locate the bar_aggregator binary relative to the project root."""
    from crypto_data_engine.common.utils.setting_utils import find_project_root

    root = find_project_root()
    name = "bar_aggregator.exe" if platform.system() == "Windows" else "bar_aggregator"
    path = root / "bin" / name
    if not path.exists():
        raise FileNotFoundError(
            f"bar_aggregator binary not found at {path}.\n"
            "Build it first:\n"
            "  cmake -B cpp/bar_aggregator/build -S cpp/bar_aggregator "
            "-DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build cpp/bar_aggregator/build"
        )
    return path


def _run_symbol(
    binary: Path,
    symbol: str,
    input_dir: str,
    output_dir: str,
    data_type: str,
    bar_type: str,
    threshold: str,
    start: str,
    end: str,
    include_advanced: bool,
    bars_per_day: int,
    lookback_days: int,
) -> int:
    """Run the C++ binary for a single symbol. Returns the exit code."""
    cmd = [
        str(binary),
        "--input-dir",   input_dir,
        "--symbol",      symbol,
        "--data-type",   data_type,
        "--start",       start,
        "--end",         end,
        "--bar-type",    bar_type,
        "--threshold",   threshold,
        "--output-dir",  output_dir,
        "--bars-per-day",  str(bars_per_day),
        "--lookback-days", str(lookback_days),
    ]
    if not include_advanced:
        cmd.append("--no-advanced")

    # Ensure Arrow/Parquet DLLs from pyarrow are discoverable at runtime.
    import pyarrow
    pyarrow_dir = str(Path(pyarrow.__file__).parent)
    pyarrow_libs = str(Path(pyarrow.__file__).parent.parent / "pyarrow.libs")
    env = os.environ.copy()
    env["PATH"] = pyarrow_dir + os.pathsep + pyarrow_libs + os.pathsep + env.get("PATH", "")

    # Stream stderr (progress) to our stderr in real time.
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, env=env)
    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
    proc.wait()
    return proc.returncode


@bar_app.command(name="aggregate", help="Aggregate tick/bookTicker data into bars")
def aggregate(
    exchange: str = typer.Option(
        "binance_futures", "--exchange", "-e", help="Exchange name"
    ),
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Symbol(s) to process. Omit to process all."
    ),
    start: str = typer.Option(..., "--start", help="Start month YYYY-MM"),
    end: str = typer.Option(..., "--end", help="End month YYYY-MM"),
    data_type: str = typer.Option(
        "aggtrades", "--data-type",
        help="Input data type: aggtrades | bookticker"
    ),
    bar_type: str = typer.Option(
        "dollar_bar", "--bar-type",
        help="Bar type: time_bar | volume_bar | dollar_bar"
    ),
    threshold: str = typer.Option(
        "auto", "--threshold",
        help="Bar threshold: numeric or 'auto' (dynamic dollar bar)"
    ),
    input_dir: Optional[str] = typer.Option(
        None, "--input-dir", help="Override input directory"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", help="Override output directory"
    ),
    include_advanced: bool = typer.Option(
        True, "--advanced/--no-advanced",
        help="Include advanced microstructure features"
    ),
    bars_per_day: int = typer.Option(
        50, "--bars-per-day", help="Target bars per day for auto threshold"
    ),
    lookback_days: int = typer.Option(
        10, "--lookback-days", help="Rolling lookback days for auto threshold"
    ),
    threads: int = typer.Option(
        1, "--threads", "-t", help="Parallel symbols (one C++ process per symbol)"
    ),
):
    from crypto_data_engine.common.config.config_settings import settings
    from crypto_data_engine.common.logger.logger import get_logger

    logger = get_logger(__name__)

    # --- Resolve binary ---
    binary = _find_binary()

    # --- Resolve input directory ---
    if input_dir is None:
        cfg = settings.downloader_cfg.get_merged_config(exchange)
        input_dir = str(cfg["data_dir"])

    # --- Resolve output directory ---
    if output_dir is None:
        from crypto_data_engine.common.config.paths import DATA_ROOT
        output_dir = str(
            DATA_ROOT / exchange / "bars" / bar_type / threshold / data_type
        )

    # --- Resolve symbol list ---
    if not symbols:
        # Enumerate all symbols from the input directory
        base = Path(input_dir)
        if data_type == "bookticker":
            base = base / "bookTicker"
        if base.exists():
            symbols = sorted(
                d.name for d in base.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
        if not symbols:
            typer.echo("[ERROR] No symbols found. Use --symbol to specify.", err=True)
            raise typer.Exit(1)

    typer.echo(
        f"[bar] {exchange} | {data_type} → {bar_type}({threshold}) | "
        f"{start}–{end} | {len(symbols)} symbol(s) | threads={threads}"
    )
    typer.echo(f"[bar] input  : {input_dir}")
    typer.echo(f"[bar] output : {output_dir}")

    # --- Run ---
    failed = []

    if threads == 1:
        for sym in symbols:
            rc = _run_symbol(
                binary, sym, input_dir, output_dir,
                data_type, bar_type, threshold, start, end,
                include_advanced, bars_per_day, lookback_days,
            )
            if rc != 0:
                failed.append(sym)
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {
                pool.submit(
                    _run_symbol,
                    binary, sym, input_dir, output_dir,
                    data_type, bar_type, threshold, start, end,
                    include_advanced, bars_per_day, lookback_days,
                ): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                rc = future.result()
                if rc != 0:
                    failed.append(sym)

    if failed:
        typer.echo(f"[ERROR] Failed symbols: {failed}", err=True)
        raise typer.Exit(1)

    typer.echo(f"[bar] Done. {len(symbols) - len(failed)}/{len(symbols)} succeeded.")
