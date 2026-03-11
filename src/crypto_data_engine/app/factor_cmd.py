"""
Factor CLI commands: compute, analyze, run, list-factors.
"""
from pathlib import Path
from typing import List, Optional

import typer

factor_app = typer.Typer(
    help="Factor computation and analysis (alphalens)",
    no_args_is_help=True,
)


def _resolve_configs(factors: Optional[List[str]]):
    """Resolve factor names to FactorConfig list."""
    from crypto_data_engine.common.config.factor_config import BUILTIN_FACTORS

    builtin = {fc.name: fc for fc in BUILTIN_FACTORS}
    if factors:
        configs = []
        for name in factors:
            if name not in builtin:
                typer.echo(f"[!] Unknown factor: {name}")
                typer.echo(f"    Available: {', '.join(sorted(builtin.keys()))}")
                raise typer.Exit(code=1)
            configs.append(builtin[name])
        return configs
    return list(builtin.values())


@factor_app.command(name="compute", help="Compute factors and save as CSV")
def compute(
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbols", "-s", help="Symbols (default: all available)"
    ),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM)"),
    end_date: str = typer.Option(..., "--end", help="End date (YYYY-MM)"),
    factors: Optional[List[str]] = typer.Option(
        None, "--factors", "-f", help="Factor names (default: all built-in)"
    ),
    interval: str = typer.Option("1h", help="Bar interval"),
    bar_type: str = typer.Option("time", help="Bar type"),
    warmup: int = typer.Option(60, help="Warmup periods to drop"),
    output_dir: str = typer.Option(
        "./data/factor_reports", "--output", "-o", help="Output directory"
    ),
):
    """
    Compute factors from bar data and save as CSV.

    Examples:
        factor compute --start 2024-06 --end 2024-12
        factor compute -s BTCUSDT -s ETHUSDT -f amihud --start 2025-01 --end 2025-03
    """
    from crypto_data_engine.services.factor_evaluator import FactorPipeline

    configs = _resolve_configs(factors)
    pipeline = FactorPipeline()

    typer.echo(f"[*] Computing {len(configs)} factors ({start_date} -> {end_date})")

    panel, price_df = pipeline.load_data(
        start_date, end_date, configs,
        symbols=symbols, warmup_periods=warmup,
        bar_type=bar_type, interval=interval,
    )
    if panel.empty:
        typer.echo("[!] No data loaded")
        raise typer.Exit(code=1)

    computed = pipeline.calculator.compute(panel, configs)
    computed, price_df = pipeline._trim_warmup(computed, price_df, start_date, warmup)

    if not computed:
        typer.echo("[!] No factors computed")
        raise typer.Exit(code=1)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, series in computed.items():
        series.to_csv(out / f"{name}.csv")
    price_df.to_csv(out / "_prices.csv")

    typer.echo(f"[+] Saved {len(computed)} factors + prices -> {out}")


@factor_app.command(name="analyze", help="Analyze pre-computed factors")
def analyze(
    input_dir: str = typer.Option(
        "./data/factor_reports", "--input", "-i", help="Directory with factor CSVs"
    ),
    output_dir: str = typer.Option(
        "./data/factor_reports", "--output", "-o", help="Output directory"
    ),
    factors: Optional[List[str]] = typer.Option(
        None, "--factors", "-f", help="Factor names (default: all in input_dir)"
    ),
    periods: str = typer.Option("1,5,10,20", help="Forward return periods"),
    quantiles: int = typer.Option(5, help="Number of quantiles"),
    max_loss: float = typer.Option(0.5, help="Max factor data loss ratio"),
    charts: bool = typer.Option(True, help="Generate PNG tear sheets"),
):
    """
    Run alphalens analysis on pre-computed factors.
    Expects factor CSVs and _prices.csv in input_dir.

    Examples:
        factor analyze
        factor analyze -f amihud -f momentum_20 --periods 1,5,10
    """
    import pandas as pd

    from crypto_data_engine.services.factor_evaluator import (
        AnalysisConfig, FactorAnalyzer, FactorReporter,
    )

    inp = Path(input_dir)
    out = Path(output_dir)

    # Load prices
    price_path = inp / "_prices.csv"
    if not price_path.exists():
        typer.echo(f"[!] Price matrix not found: {price_path}")
        typer.echo("    Run 'factor compute' first.")
        raise typer.Exit(code=1)
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    typer.echo(f"[*] Loaded prices: {price_df.shape}")

    # Discover factor CSVs
    if factors:
        csv_files = [(inp / f"{name}.csv") for name in factors]
        missing = [f for f in csv_files if not f.exists()]
        if missing:
            typer.echo(f"[!] Not found: {[m.name for m in missing]}")
            raise typer.Exit(code=1)
    else:
        csv_files = sorted(
            f for f in inp.glob("*.csv")
            if f.name not in ("_prices.csv", "summary.csv")
            and f.parent == inp
        )

    if not csv_files:
        typer.echo("[!] No factor CSVs found")
        raise typer.Exit(code=1)

    # Load factors
    factor_series = {}
    for fpath in csv_files:
        if fpath.parent != inp:
            continue
        try:
            s = pd.read_csv(fpath, index_col=[0, 1], parse_dates=True).iloc[:, 0]
            s.name = fpath.stem
            factor_series[fpath.stem] = s
        except Exception as exc:
            typer.echo(f"[!] Failed to load {fpath.name}: {exc}")

    typer.echo(f"[*] Loaded {len(factor_series)} factors")

    # Analyze
    parsed_periods = tuple(int(p.strip()) for p in periods.split(","))
    analyzer = FactorAnalyzer(AnalysisConfig(
        periods=parsed_periods, quantiles=quantiles, max_loss=max_loss,
    ))
    reporter = FactorReporter()

    batch_results = analyzer.analyze_batch(factor_series, price_df)

    if not batch_results:
        typer.echo("[!] No factors analyzed successfully")
        raise typer.Exit(code=1)

    # Report
    reporter.export(batch_results, out, charts=charts)
    summary = reporter.summary_table(batch_results)
    typer.echo("\n=== Factor Summary ===")
    typer.echo(summary.to_string())
    typer.echo(f"\n[+] Results -> {out}")


@factor_app.command(name="run", help="Compute + analyze in one step")
def run(
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbols", "-s", help="Symbols (default: all available)"
    ),
    start_date: str = typer.Option(..., "--start", help="Start date (YYYY-MM)"),
    end_date: str = typer.Option(..., "--end", help="End date (YYYY-MM)"),
    factors: Optional[List[str]] = typer.Option(
        None, "--factors", "-f", help="Factor names (default: all built-in)"
    ),
    interval: str = typer.Option("1h", help="Bar interval"),
    bar_type: str = typer.Option("time", help="Bar type"),
    warmup: int = typer.Option(60, help="Warmup periods"),
    periods: str = typer.Option("1,5,10,20", help="Forward return periods"),
    quantiles: int = typer.Option(5, help="Number of quantiles"),
    output_dir: str = typer.Option(
        "./data/factor_reports", "--output", "-o", help="Output directory"
    ),
    charts: bool = typer.Option(True, help="Generate PNG tear sheets"),
):
    """
    End-to-end: load → compute → analyze → export.

    Examples:
        factor run --start 2024-06 --end 2024-12
        factor run -s BTCUSDT -s ETHUSDT -f amihud --start 2025-01 --end 2025-03
    """
    from crypto_data_engine.services.factor_evaluator import (
        AnalysisConfig, FactorAnalyzer, FactorPipeline, FactorReporter,
    )

    configs = _resolve_configs(factors)
    parsed_periods = tuple(int(p.strip()) for p in periods.split(","))

    pipeline = FactorPipeline(
        analyzer=FactorAnalyzer(AnalysisConfig(
            periods=parsed_periods, quantiles=quantiles,
        )),
        reporter=FactorReporter(),
    )

    typer.echo(f"[*] Running {len(configs)} factors ({start_date} -> {end_date})")

    batch_results, summary = pipeline.run(
        start_date=start_date,
        end_date=end_date,
        factor_configs=configs,
        symbols=symbols,
        warmup_periods=warmup,
        bar_type=bar_type,
        interval=interval,
        output_dir=Path(output_dir),
        charts=charts,
    )

    if not batch_results:
        typer.echo("[!] Pipeline failed")
        raise typer.Exit(code=1)

    typer.echo("\n=== Factor Summary ===")
    typer.echo(summary.to_string())
    typer.echo(f"\n[+] Done! Results -> {output_dir}")


@factor_app.command(name="list-factors", help="List all built-in factors")
def list_factors():
    """Show all built-in factor definitions."""
    from crypto_data_engine.common.config.factor_config import (
        BUILTIN_FACTORS, FactorType,
    )

    configs = BUILTIN_FACTORS
    typer.echo(f"Built-in factors ({len(configs)}):\n")
    typer.echo(f"{'Name':<25} {'Type':<10} {'Column':<30} {'Details'}")
    typer.echo("-" * 85)
    for fc in configs:
        details = ""
        if fc.factor_type == FactorType.ROLLING:
            details = f"window={fc.window}, method={fc.rolling_method.value}"
        typer.echo(
            f"{fc.name:<25} {fc.factor_type.value:<10} {(fc.column or '-'):<30} {details}"
        )
