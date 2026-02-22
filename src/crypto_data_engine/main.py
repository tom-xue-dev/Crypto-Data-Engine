"""
Crypto Data Engine - Quantitative Trading CLI

Top-level entry point. Registers sub-commands from app/ modules.

Usage:
    crypto-engine --help
    crypto-engine serve
    crypto-engine dev
    crypto-engine data download --start-date 2025-01 --end-date 2025-06
    crypto-engine data list
    crypto-engine aggregate BTCUSDT --bar-type dollar_bar
    crypto-engine features data.parquet
    crypto-engine backtest --strategy momentum
    crypto-engine pipeline run --top-n 100 --workers 8
    crypto-engine init
    crypto-engine test
"""
import os
import subprocess
import sys

import typer

from crypto_data_engine.app.server import server_app
from crypto_data_engine.app.data_cmd import data_app
from crypto_data_engine.app.aggregate_cmd import aggregate_app
from crypto_data_engine.app.feature_cmd import feature_app
from crypto_data_engine.app.backtest_cmd import backtest_app
from crypto_data_engine.app.pipeline_cmd import pipeline_app
from crypto_data_engine.app.enrich_cmd import enrich_app

app = typer.Typer(
    help="Crypto Data Engine - Quantitative Trading CLI",
    no_args_is_help=True,
)

# Register server commands (serve, dev) at root level
app.add_typer(server_app, name="", help="")

# Register sub-command groups
app.add_typer(data_app, name="data")
app.add_typer(aggregate_app, name="aggregate")
app.add_typer(feature_app, name="features")
app.add_typer(backtest_app, name="backtest")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(enrich_app, name="enrich")


# ============================================================================
# Utility commands (kept at root level)
# ============================================================================

@app.command(help="Initialize YAML config templates")
def init():
    """Generate all YAML template files."""
    from crypto_data_engine.common.config.config_settings import create_all_templates

    typer.echo("[*] Initializing YAML templates...")
    create_all_templates()
    typer.echo("[+] Config templates created")


@app.command(help="Run tests for the project")
def test(
    file: str = typer.Option("", help="Specific test file (e.g. test_trading_log.py)"),
    verbose: bool = typer.Option(True, help="Verbose output"),
    coverage: bool = typer.Option(False, help="Generate coverage report"),
    quick: bool = typer.Option(False, help="Run only quick tests"),
):
    """Run pytest tests."""
    cmd = [sys.executable, "-m", "pytest"]

    if file:
        cmd.append(f"tests/{file}")
    else:
        cmd.append("tests/")

    if verbose:
        cmd.append("-v")

    if quick:
        cmd.extend(["-x", "--tb=short"])

    if coverage:
        cmd.extend(["--cov=crypto_data_engine", "--cov-report=html"])

    typer.echo(f"[*] Running tests: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    os.environ["PYTHONPATH"] = os.path.abspath("src")
    app()


if __name__ == "__main__":
    main()
