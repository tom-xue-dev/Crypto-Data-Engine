import logging
import sys
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import subprocess

import ray

from common.config.config_settings import settings
from common.logger.logger import logger, setup_logger
from server.routers.datascraper import tick_router




@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("🛑 Shutting down…")
    try:
        ray.shutdown()
    except Exception:
        logger.warning("Ray shutdown failed or Ray not installed")


def register_routers(app: FastAPI) -> None:
    app.include_router(tick_router)


def create_app() -> FastAPI:
    app = FastAPI(
        title="BTC Trading Platform",
        version="0.1.0",
        docs_url="/docs",
        lifespan=lifespan
    )
    register_routers(app)
    return app


def server_startup(host:int = None,port:int = None):
    setup_logger()
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

