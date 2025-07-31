from datetime import datetime

from loguru import logger
import sys
import logging
from common.config.paths import DATA_DIR
LOG_PATH = DATA_DIR/"logs"/datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def setup_logger():
    logger.remove()  # 清空默认配置
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        enqueue=True  # 支持多线程/多进程安全
    )
    logger.add(
        LOG_PATH,
        level="DEBUG",
        rotation="10 MB",        # 每 10MB 切分
        retention="7 days",      # 保留 7 天
        compression="zip",       # 压缩旧日志
        encoding="utf-8",
        enqueue=True
    )
    # 接管标准库日志（requests、uvicorn、httpx、FastAPI等）
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except Exception:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
    for name in ["uvicorn", "uvicorn.error", "httpx", "requests"]:
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False

    logger.info("🚀 Logger initialized")



