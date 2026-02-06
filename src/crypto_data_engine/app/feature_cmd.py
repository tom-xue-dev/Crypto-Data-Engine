"""
Feature calculation CLI command.
"""
from pathlib import Path

import pandas as pd
import typer

feature_app = typer.Typer(help="Calculate features from bar data")


@feature_app.callback(invoke_without_command=True)
def features(
    input_file: str = typer.Argument(..., help="Input bar data file (parquet)"),
    output_file: str = typer.Option(None, help="Output file (default: input_features.parquet)"),
    windows: str = typer.Option("5,10,20,60,120", help="Rolling windows (comma-separated)"),
    include_alphas: bool = typer.Option(True, help="Include alpha factors"),
    include_technical: bool = typer.Option(False, help="Include technical indicators (requires talib)"),
    normalize: bool = typer.Option(False, help="Normalize features"),
):
    """
    Calculate features from bar data using the unified feature calculator.

    Examples:
        features data/bars/BTCUSDT/BTCUSDT_dollar_bar_1000000.parquet
        features data.parquet --windows 5,10,20
        features data.parquet --normalize
    """
    from crypto_data_engine.services.feature import (
        UnifiedFeatureConfig,
        UnifiedFeatureCalculator,
    )

    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"[!] Input file not found: {input_file}")
        raise typer.Exit(code=1)

    window_list = [int(w.strip()) for w in windows.split(",")]

    typer.echo(f"[*] Calculating features for {input_file}")
    typer.echo(f"[*] Windows: {window_list}")

    try:
        bars = pd.read_parquet(input_path)
        typer.echo(f"[+] Loaded {len(bars)} bars")
    except Exception as error:
        typer.echo(f"[!] Error loading file: {error}")
        raise typer.Exit(code=1)

    config = UnifiedFeatureConfig(
        windows=window_list,
        include_returns=True,
        include_volatility=True,
        include_momentum=True,
        include_volume=True,
        include_microstructure=True,
        include_alphas=include_alphas,
        include_technical=include_technical,
        normalize=normalize,
    )

    calculator = UnifiedFeatureCalculator(config)

    try:
        feature_df = calculator.calculate(bars)

        feature_count = len(feature_df.columns) - len(bars.columns)
        typer.echo(f"[+] Calculated {feature_count} features")

        if output_file is None:
            output_file = input_path.stem + "_features.parquet"

        output_path = Path(output_file)
        feature_df.to_parquet(output_path)

        typer.echo(f"[+] Saved to {output_path}")

        # Show feature summary
        typer.echo("\nFeature Groups:")
        feature_cols = [c for c in feature_df.columns if c not in bars.columns]
        groups = {}
        for col in feature_cols:
            prefix = col.split("_")[0]
            groups[prefix] = groups.get(prefix, 0) + 1

        for group, count in sorted(groups.items(), key=lambda x: -x[1])[:10]:
            typer.echo(f"  {group}: {count} features")

    except Exception as error:
        typer.echo(f"[!] Feature calculation failed: {error}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
