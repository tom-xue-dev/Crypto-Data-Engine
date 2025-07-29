import logging
import sys
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import subprocess

import ray
from common.config.load_config import server_cfg
from server.routers.datascraper import data_scraper_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("🛑 Shutting down…")
    try:
        ray.shutdown()
    except Exception:
        logger.warning("Ray shutdown failed or Ray not installed")


def register_routers(app: FastAPI) -> None:
    app.include_router(data_scraper_router)


def create_app() -> FastAPI:
    app = FastAPI(
        title="BTC Trading Platform",
        version="0.1.0",
        docs_url="/docs",
        lifespan=lifespan
    )
    register_routers(app)
    return app


def server_startup():
    cfg = server_cfg
    logger.info(f"🚀 Starting API server at {cfg.host}:{cfg.port}")
    app = create_app()
    uvicorn.run(app, host=cfg.host, port=cfg.port)


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


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python startup.py [api|worker <type>]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "api":
        server_startup()
    elif mode == "worker":
        if len(sys.argv) != 3:
            logger.error("Usage: python startup.py worker [download|preprocess|backtest]")
            sys.exit(1)
        start_worker(sys.argv[2])
    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
