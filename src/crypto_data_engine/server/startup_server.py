import sys
from fastapi import FastAPI
import uvicorn
import subprocess
from crypto_data_engine.common.logger.logger import get_logger, setup_logger
logger = get_logger(__name__)

def register_routers(app: FastAPI) -> None:
    """
    register your routers here
    """
    from crypto_data_engine.server.routers.datascraper import datascraper_router
    routers = [
        ("tick", datascraper_router),
    ]

    total = 0
    for name, router in routers:
        app.include_router(router)
        logger.info(f"name={name} tags={router.tags or []}")
        total += 1

    logger.info(f"✅ Routers registered: {len(routers)}, routes total: {total}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="BTC Trading Platform",
        version="0.1.0",
        docs_url="/docs",
    )
    register_routers(app)
    return app


def server_startup(host= None,port:int = None):
    setup_logger()
    from crypto_data_engine.common.config.config_settings import settings
    cfg = settings.server_cfg
    host = cfg.host if host is None else host
    port = cfg.port if port is None else port
    logger.info(f"🚀 Starting API server at {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port)


def start_worker(service: str):
    worker_name_map = {
        "download": "download_tasks",
        "preprocess": "bar_tasks",
        "backtest": "backtest_tasks",
    }
    queue = worker_name_map.get(service)
    if not queue:
        logger.error(f"❌ Unknown service '{service}'")
        sys.exit(1)

    cmd = [
        "celery", "-A", "celery_app.celery_app", "worker",
        "--loglevel=info",
        f"--queues={queue}",
        f"--hostname={service}@%h"
    ]
    logger.info(f"🚀 Starting Celery worker: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    server_startup(None, None)
