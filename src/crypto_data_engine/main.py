"""
Crypto Data Engine - CLI entry point.

Usage:
    poetry run main factor run --start 2024-06 --end 2024-12
    poetry run main factor list-factors
    poetry run main data list
"""
import os
import subprocess
import sys

import typer

from crypto_data_engine.app.bar_cmd import bar_app
from crypto_data_engine.app.data_cmd import data_app
from crypto_data_engine.app.factor_cmd import factor_app

app = typer.Typer(
    help="Crypto Data Engine - Quantitative Trading CLI",
    no_args_is_help=True,
)

app.add_typer(data_app, name="data")
app.add_typer(bar_app, name="bar")
app.add_typer(factor_app, name="factor")


@app.command(help="Initialize YAML config templates")
def init():
    from crypto_data_engine.common.config.config_settings import create_all_templates
    typer.echo("[*] Initializing YAML templates...")
    create_all_templates()
    typer.echo("[+] Config templates created")


@app.command(help="Run tests for the project")
def test(
    file: str = typer.Option("", help="Specific test file"),
    verbose: bool = typer.Option(True, help="Verbose output"),
    coverage: bool = typer.Option(False, help="Generate coverage report"),
    quick: bool = typer.Option(False, help="Run only quick tests"),
):
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
