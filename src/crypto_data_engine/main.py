import typer
import subprocess
import os

app = typer.Typer(help="🧠 Quant System COMMANDS")

@app.command(help="start up fastapi server")
def start(host = None,port: int = None):
    """启动 FastAPI 网关服务"""
    from crypto_data_engine.server.startup_server import server_startup
    server_startup(host,port)


@app.command(help = "start celery worker to consume tasks")
def run_worker(module: str = typer.Argument(..., help="模块名: downloader / bar_generator / backtest_engine")):
    """启动指定模块的 Celery worker"""
    typer.echo(f"🎯 启动 {module} 的 Celery Worker")
    worker_module = f"{module}.tasks"
    subprocess.run(["celery", "-A", "task_manager.celery_app", "worker", "--loglevel=info","--pool=solo"])

@app.command(help= "initialize all yaml template files")
def init_config():
    from crypto_data_engine.common.config.config_settings import create_all_templates
    typer.echo(f"initializing all yaml template files...")
    create_all_templates()

@app.command(help = "init db")
def init_db():
    from crypto_data_engine.db.db_init import init
    typer.echo(f"starting init db...")
    try:
        init()
    except Exception as e:
        typer.echo(f"init db failed: {str(e)}")
@app.command()
def dev_all():
    """开发环境下同时启动所有核心服务"""
    typer.echo("🌈 启动 downloader / bar_generator / backtest_engine worker 以及 FastAPI...")
    cmds = [
        ["celery", "-A", "downloader.tasks", "worker", "--loglevel=info"],
        ["celery", "-A", "bar_generator.tasks", "worker", "--loglevel=info"],
        ["celery", "-A", "backtest_engine.tasks", "worker", "--loglevel=info"],
        ["uvicorn", "api_gateway.main:app", "--reload", "--port", "8000"]
    ]
    for cmd in cmds:
        subprocess.Popen(cmd)
    typer.echo("✅ 所有服务已在后台启动")

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
def logs(service: str = typer.Argument(..., help="Docker 服务名，如 api / downloader")):
    """查看某个服务日志"""
    subprocess.run(["docker-compose", "logs", "-f", service])


def main():
    os.environ["PYTHONPATH"] = os.path.abspath("src")
    app()

if __name__ == "__main__":
    main()
