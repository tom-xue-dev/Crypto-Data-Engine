import typer
import subprocess
import os

app = typer.Typer(help="🧠 Quant System Commands")

@app.command(help="Start FastAPI server")
def start(host = None,port: int = None):
    """Start the FastAPI gateway service."""
    from crypto_data_engine.server.startup_server import server_startup
    server_startup(host,port)


@app.command(help = "Start Celery worker to consume tasks")
def run_worker(module: str = typer.Argument(..., help="Module name: downloader / bar_generator / backtest_engine")):
    """Launch Celery worker for the specified module."""
    typer.echo(f"🎯 Launching Celery worker for {module}")
    worker_module = f"{module}.tasks"
    subprocess.run(["celery", "-A", "task_manager.celery_app", "worker", "--loglevel=info","--pool=solo"])

@app.command(help= "Initialize all YAML template files")
def init_config():
    from crypto_data_engine.common.config.config_settings import create_all_templates
    typer.echo(f"Initializing YAML templates...")
    create_all_templates()

@app.command(help = "Initialize database")
def init_db():
    from crypto_data_engine.db.db_init import init
    typer.echo(f"Starting database initialization...")
    try:
        init()
    except Exception as e:
        typer.echo(f"Database init failed: {str(e)}")
@app.command()
def dev_all():
    """Start all core services in development mode."""
    typer.echo("🌈 Launching downloader / bar_generator / backtest_engine workers and FastAPI...")
    cmds = [
        ["celery", "-A", "downloader.tasks", "worker", "--loglevel=info"],
        ["celery", "-A", "bar_generator.tasks", "worker", "--loglevel=info"],
        ["celery", "-A", "backtest_engine.tasks", "worker", "--loglevel=info"],
        ["uvicorn", "api_gateway.main:app", "--reload", "--port", "8000"]
    ]
    for cmd in cmds:
        subprocess.Popen(cmd)
    typer.echo("✅ All services started in background")

# @app.command()
# def deploy():
#     """使用 Docker Compose 启动全部服务"""
#     typer.echo("🐳 正在部署 Docker Compose 服务...")
#     subprocess.run(["docker-compose", "up", "-d", "--build"])
#
# @app.command()
# def status():
#     """查看当前服务状态"""
#     typer.echo("🔍 检查容器状态")
#     subprocess.run(["docker-compose", "ps"])

@app.command()
def logs(service: str = typer.Argument(..., help="Docker service name, e.g. api / downloader")):
    """Tail logs for a specific Docker service."""
    subprocess.run(["docker-compose", "logs", "-f", service])


def main():
    os.environ["PYTHONPATH"] = os.path.abspath("src")
    app()

if __name__ == "__main__":
    main()
