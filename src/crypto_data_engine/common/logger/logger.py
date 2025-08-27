"""
统一日志配置管理模块 - 改进版
支持：模块化logger、单例配置、性能优化
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger as _loguru_logger
from pydantic import BaseModel
from crypto_data_engine.common.config.paths import PROJECT_ROOT


class LogConfig(BaseModel):
    """日志配置模型"""
    level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_file: str = "app.log"
    rotation: str = "10 MB"
    retention: str = "7 days"
    compression: str = "zip"
    enqueue: bool = True
    intercept_standard_logging: bool = True
    format_console: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]}</cyan> - "
        "<level>{message}</level>"
    )
    format_file: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[module]} | "
        "{process}:{thread} | "
        "{message}"
    )


class LoggerManager:
    """日志管理器 - 单例模式"""

    _instance: Optional['LoggerManager'] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[LogConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[LogConfig] = None):
        if not hasattr(self, 'config'):
            self.config = config or LogConfig()
            self._module_loggers: Dict[str, Any] = {}

    def setup_logger(self) -> None:
        """全局日志配置 - 只执行一次"""
        if self._initialized:
            return

        # 清空默认配置
        _loguru_logger.remove()

        # 确保日志目录存在
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = self.config.log_dir / self.config.log_file

        # 控制台输出
        _loguru_logger.add(
            sys.stdout,
            level=self.config.console_level,
            format=self.config.format_console,
            enqueue=self.config.enqueue,
            colorize=True
        )

        # 文件输出
        _loguru_logger.add(
            str(log_file_path),
            level=self.config.file_level,
            format=self.config.format_file,
            rotation=self.config.rotation,
            retention=self.config.retention,
            compression=self.config.compression,
            encoding="utf-8",
            enqueue=self.config.enqueue
        )

        # 接管标准库日志
        if self.config.intercept_standard_logging:
            self._setup_standard_logging_intercept()

        self._initialized = True
        _loguru_logger.bind(module="LoggerManager").info("🚀 Logger initialized successfully")
        _loguru_logger.bind(module="LoggerManager").info(f"📁 Log files -> {log_file_path}")

    def get_module_logger(self, module_name: str):
        """获取特定模块的 logger - 缓存复用"""
        if module_name not in self._module_loggers:
            # 确保全局配置已初始化
            if not self._initialized:
                self.setup_logger()

            # 创建绑定模块名的 logger
            self._module_loggers[module_name] = _loguru_logger.bind(module=module_name)

        return self._module_loggers[module_name]

    def _setup_standard_logging_intercept(self) -> None:
        """接管标准库日志"""
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = _loguru_logger.level(record.levelname).name
                except Exception:
                    level = record.levelno

                # 使用原始 logger 名作为模块标识
                bound_logger = _loguru_logger.bind(module=record.name)
                bound_logger.opt(depth=6, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        logging.basicConfig(
            handlers=[InterceptHandler()],
            level=logging.INFO,
            force=True
        )

        # 接管常用库
        intercepted_loggers = [
            "uvicorn", "uvicorn.error", "uvicorn.access",
            "fastapi", "httpx", "requests",
            "celery", "celery.worker", "celery.task",
            "redis", "sqlalchemy"
        ]

        for logger_name in intercepted_loggers:
            logging.getLogger(logger_name).handlers = [InterceptHandler()]
            logging.getLogger(logger_name).propagate = False

    def add_service_handler(self, service_name: str) -> None:
        """为特定服务添加专门的日志文件"""
        service_log_file = self.config.log_dir / f"{service_name}.log"

        _loguru_logger.add(
            str(service_log_file),
            level=self.config.file_level,
            format=self.config.format_file,
            filter=lambda record: record["extra"].get("service") == service_name,
            rotation=self.config.rotation,
            retention=self.config.retention,
            compression=self.config.compression,
            encoding="utf-8",
            enqueue=self.config.enqueue
        )

        _loguru_logger.bind(module="LoggerManager").info(
            f"📝 Service handler added: '{service_name}' -> {service_log_file}"
        )


# 全局管理器实例
_manager: Optional[LoggerManager] = None


def get_logger_manager(config: Optional[LogConfig] = None) -> LoggerManager:
    """获取全局日志管理器单例"""
    global _manager
    if _manager is None:
        _manager = LoggerManager(config)
    return _manager


def setup_logger(config: Optional[LogConfig] = None) -> None:
    """全局日志初始化 - 应用启动时调用一次"""
    manager = get_logger_manager(config)
    manager.setup_logger()


def get_logger(module_name: Optional[str] = None):
    """获取模块专用 logger - 每个模块调用一次并缓存"""
    if module_name is None:
        # 获取调用者的模块名
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            module_name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            module_name = 'unknown'

    manager = get_logger_manager()
    return manager.get_module_logger(module_name)


def get_service_logger(service_name: str):
    """获取服务专用 logger - 会写入独立文件"""
    manager = get_logger_manager()
    manager.add_service_handler(service_name)
    return _loguru_logger.bind(module=service_name, service=service_name)