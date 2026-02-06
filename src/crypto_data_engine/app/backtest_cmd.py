"""
Backtest CLI command.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

backtest_app = typer.Typer(help="Run backtests")


@backtest_app.callback(invoke_without_command=True)
def backtest(
    strategy: str = typer.Option("momentum", help="Strategy: momentum, mean_reversion, equal_weight, long_short"),
    mode: str = typer.Option("cross_sectional", help="Mode: cross_sectional, time_series"),
    capital: float = typer.Option(1000000, help="Initial capital"),
    data_file: str = typer.Option(None, help="Input data file (parquet with features)"),
    start_date: str = typer.Option("2024-01-01", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2024-06-30", help="End date (YYYY-MM-DD)"),
    rebalance: str = typer.Option("W", help="Rebalance frequency (D, W, M)"),
    commission: float = typer.Option(0.001, help="Commission rate"),
    slippage: float = typer.Option(0.0005, help="Slippage rate"),
    max_position: float = typer.Option(0.2, help="Max position size (0-1)"),
    output_dir: str = typer.Option("./data/backtest_results", help="Output directory"),
):
    """
    Run a backtest from command line and save results.

    Examples:
        backtest --strategy momentum --data-file data/features.parquet
        backtest --strategy long_short --capital 500000 --commission 0.0005
        backtest --mode time_series --strategy momentum_ts
    """
    from crypto_data_engine.services.back_test import (
        BacktestConfig,
        BacktestMode,
        RiskConfigModel,
        CostConfigModel,
        create_backtest_engine,
        create_strategy,
    )

    import numpy as np

    typer.echo(f"[*] Running {strategy} strategy in {mode} mode")
    typer.echo(f"[*] Date range: {start_date} to {end_date}")
    typer.echo(f"[*] Capital: ${capital:,.0f}")

    mode_map = {
        "cross_sectional": BacktestMode.CROSS_SECTIONAL,
        "time_series": BacktestMode.TIME_SERIES,
    }
    backtest_mode = mode_map.get(mode, BacktestMode.CROSS_SECTIONAL)

    config = BacktestConfig(
        mode=backtest_mode,
        initial_capital=capital,
        start_date=datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc),
        end_date=datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc),
        rebalance_frequency=rebalance,
        warmup_periods=30,
        risk_config=RiskConfigModel(
            max_position_size=max_position,
            max_leverage=1.0,
        ),
        cost_config=CostConfigModel(
            commission_rate=commission,
            slippage_rate=slippage,
        ),
    )

    try:
        if mode == "cross_sectional":
            strategy_obj = create_strategy(strategy, lookback_col="return_20", top_n_long=10, top_n_short=10)
        else:
            strategy_obj = create_strategy(f"{strategy}_ts", momentum_column="momentum_20", long_threshold=0.02)
    except Exception as error:
        typer.echo(f"[!] Failed to create strategy: {error}")
        raise typer.Exit(code=1)

    engine = create_backtest_engine(config, strategy_obj)

    if data_file and Path(data_file).exists():
        typer.echo(f"[*] Loading data from {data_file}...")
        data = pd.read_parquet(data_file)
        typer.echo(f"[+] Loaded {len(data)} rows")
    else:
        typer.echo("[*] No data file provided, generating demo data...")
        np.random.seed(42)

        n_days = 120
        n_assets = 10

        data_list = []
        for i in range(n_assets):
            asset = f"ASSET{i+1}"
            dates = pd.date_range(start_date, periods=n_days, freq="D", tz=timezone.utc)

            returns = np.random.randn(n_days) * 0.02
            prices = 100 * np.exp(np.cumsum(returns))

            df = pd.DataFrame({
                "timestamp": dates,
                "asset": asset,
                "close": prices,
                "volume": np.random.exponential(1000, n_days),
                "return_20": pd.Series(prices).pct_change(20).values,
                "momentum_20": pd.Series(prices).pct_change(20).values,
            })
            data_list.append(df)

        data = pd.concat(data_list, ignore_index=True)
        data = data.set_index(["timestamp", "asset"])
        typer.echo(f"[+] Generated demo data: {len(data)} rows, {n_assets} assets")

    typer.echo("[*] Running backtest...")
    try:
        result = engine.run(data)

        typer.echo("\n" + "=" * 60)
        typer.echo("Backtest Results")
        typer.echo("=" * 60)
        typer.echo(f"  Strategy:      {strategy}")
        typer.echo(f"  Mode:          {mode}")
        typer.echo(f"  Initial:       ${result.initial_capital:,.2f}")
        typer.echo(f"  Final:         ${result.final_capital:,.2f}")
        typer.echo("-" * 60)
        typer.echo(f"  Total Return:  {result.total_return*100:+.2f}%")
        typer.echo(f"  Annual Return: {result.annual_return*100:+.2f}%")
        typer.echo(f"  Sharpe Ratio:  {result.sharpe_ratio:.3f}")
        typer.echo(f"  Max Drawdown:  {result.max_drawdown*100:.2f}%")
        typer.echo(f"  Total Trades:  {result.total_trades}")
        typer.echo(f"  Win Rate:      {result.win_rate*100:.1f}%")
        typer.echo("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        task_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        result_file = output_path / f"result_{task_id}.json"
        result_dict = {
            "task_id": task_id,
            "strategy": strategy,
            "mode": mode,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
        }
        with open(result_file, "w") as file_handle:
            json.dump(result_dict, file_handle, indent=2)
        typer.echo(f"\n[+] Results saved to {result_file}")

    except Exception as error:
        typer.echo(f"[!] Backtest failed: {error}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
