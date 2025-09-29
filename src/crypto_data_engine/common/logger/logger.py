"""
Unified logging configuration module (enhanced).
Supports modular loggers, singleton configuration, and performance tuning.
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger as _loguru_logger
from pydantic import BaseModel
from crypto_data_engine.common.config.paths import PROJECT_ROOT


class LogConfig(BaseModel):
    """Logging configuration model."""
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
    """Singleton logger manager."""

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
        """Configure global logging once."""
        if self._initialized:
            return

        # Clear default sinks
        _loguru_logger.remove()

        # Ensure log directory exists
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = self.config.log_dir / self.config.log_file

        # Console sink
        _loguru_logger.add(
            sys.stdout,
            level=self.config.console_level,
            format=self.config.format_console,
            enqueue=self.config.enqueue,
            colorize=True
        )

        # File sink
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

        # Intercept standard logging if enabled
        if self.config.intercept_standard_logging:
            self._setup_standard_logging_intercept()

        self._initialized = True
        _loguru_logger.bind(module="LoggerManager").info("🚀 Logger initialized successfully")
        _loguru_logger.bind(module="LoggerManager").info(f"📁 Log files -> {log_file_path}")

    def get_module_logger(self, module_name: str):
        """Return module-specific logger with caching."""
        if module_name not in self._module_loggers:
            # Ensure global configuration initialized
            if not self._initialized:
                self.setup_logger()

            # Create logger bound to module name
            self._module_loggers[module_name] = _loguru_logger.bind(module=module_name)

        return self._module_loggers[module_name]

    def _setup_standard_logging_intercept(self) -> None:
        """Intercept standard library logging and redirect to Loguru."""
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = _loguru_logger.level(record.levelname).name
                except Exception:
                    level = record.levelno

                # Use original logger name as module marker
                bound_logger = _loguru_logger.bind(module=record.name)
                bound_logger.opt(depth=6, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        logging.basicConfig(
            handlers=[InterceptHandler()],
            level=logging.INFO,
            force=True
        )

        # Intercept common libraries
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
        """Attach dedicated log file for a given service."""
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


# Global manager instance
_manager: Optional[LoggerManager] = None


def get_logger_manager(config: Optional[LogConfig] = None) -> LoggerManager:
    """Return singleton logger manager."""
    global _manager
    if _manager is None:
        _manager = LoggerManager(config)
    return _manager


def setup_logger(config: Optional[LogConfig] = None) -> None:
    """Initialize logging once at application startup."""
    manager = get_logger_manager(config)
    manager.setup_logger()


def get_logger(module_name: Optional[str] = None):
    """Get module-specific logger and cache it."""
    if module_name is None:
        # Infer caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            module_name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            module_name = 'unknown'

    manager = get_logger_manager()
    return manager.get_module_logger(module_name)


def get_service_logger(service_name: str):
    """Get service-specific logger writing to dedicated file."""
    manager = get_logger_manager()
    manager.add_service_handler(service_name)
    return _loguru_logger.bind(module=service_name, service=service_name)