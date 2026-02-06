"""
Server CLI commands: serve and dev.
"""
import subprocess
import sys

import typer

server_app = typer.Typer(help="API server commands")


@server_app.command(help="Start API server")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the FastAPI server."""
    typer.echo(f"[*] Starting API server on http://{host}:{port}")
    typer.echo(f"[*] API docs available at http://{host}:{port}/docs")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "crypto_data_engine.api.main:app",
        "--host", host,
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    subprocess.run(cmd)


@server_app.command(help="Start API server in development mode (with auto-reload)")
def dev(
    port: int = typer.Option(8000, help="API server port"),
):
    """Start API server with auto-reload enabled."""
    serve(host="127.0.0.1", port=port, reload=True)
