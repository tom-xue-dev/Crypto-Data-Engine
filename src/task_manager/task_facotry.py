from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from celery import Celery
import logging

from task_manager.config import TaskType, TaskConfig

logger = logging.getLogger(__name__)



class TaskHandler(ABC):
    """Abstract base class for task handlers."""

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute core task logic."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate task configuration."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Task type."""
        pass


class TaskFactory:
    """Task factory managing registration and creation."""

    def __init__(self):
        self._handlers: Dict[TaskType, TaskHandler] = {}
        self._task_configs: Dict[TaskType, TaskConfig] = {}

    def register_handler(self, handler: TaskHandler, config: TaskConfig) -> None:
        """Register task handler."""
        task_type = handler.task_type

        if task_type in self._handlers:
            logger.warning(f"Overriding existing handler: {task_type}")

        self._handlers[task_type] = handler
        self._task_configs[task_type] = config
        logger.info(f"Registered handler: {task_type.value}")

    def get_handler(self, task_type: TaskType) -> Optional[TaskHandler]:
        """Fetch task handler."""
        return self._handlers.get(task_type)

    def get_config(self, task_type: TaskType) -> Optional[TaskConfig]:
        """Fetch task configuration."""
        return self._task_configs.get(task_type)

    def list_registered_tasks(self) -> List[TaskType]:
        """List registered task types."""
        return list(self._handlers.keys())

    def create_celery_tasks(self, celery_app: Celery) -> None:
        """Create Celery tasks for all registered handlers."""
        for task_type, handler in self._handlers.items():
            config = self._task_configs[task_type]
            self._create_single_task(celery_app, task_type, handler, config)

    def _create_single_task(self, celery_app: Celery, task_type: TaskType,
                            handler: TaskHandler, config: TaskConfig) -> None:
        """Create single Celery task."""

        def task_wrapper(**kwargs):
            """Task wrapper adding generic logic."""
            try:
                # Validate configuration
                if not handler.validate_config(kwargs):
                    return {
                        "status": "FAILED",
                        "error": "Invalid task configuration",
                        "task_type": task_type.value
                    }

                # Execute task
                result = handler.execute(**kwargs)

                # Attach metadata
                result.update({
                    "task_type": task_type.value,
                    "timestamp": __import__("time").time()
                })

                return result

            except Exception as e:
                logger.error(f"Task execution failed {task_type.value}: {str(e)}")
                return {
                    "status": "FAILED",
                    "error": str(e),
                    "task_type": task_type.value
                }

        # Configure task
        task_options = {
            "name": config.name,
            "bind": True,
            "queue": config.queue,
        }

        if config.time_limit:
            task_options["time_limit"] = config.time_limit
        if config.soft_time_limit:
            task_options["soft_time_limit"] = config.soft_time_limit
        if config.retry_policy:
            task_options.update(config.retry_policy)

        # Register with Celery
        celery_app.task(**task_options)(task_wrapper)
        logger.info(f"Created Celery task: {config.name}")


# Global task factory instance
task_factory = TaskFactory()