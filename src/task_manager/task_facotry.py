from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from celery import Celery
import logging

from task_manager.config import TaskType, TaskConfig

logger = logging.getLogger(__name__)



class TaskHandler(ABC):
    """任务处理器抽象基类"""

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行任务的核心逻辑"""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证任务配置"""
        pass

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """任务类型"""
        pass


class TaskFactory:
    """任务工厂 - 管理任务的注册和创建"""

    def __init__(self):
        self._handlers: Dict[TaskType, TaskHandler] = {}
        self._task_configs: Dict[TaskType, TaskConfig] = {}

    def register_handler(self, handler: TaskHandler, config: TaskConfig) -> None:
        """注册任务处理器"""
        task_type = handler.task_type

        if task_type in self._handlers:
            logger.warning(f"覆盖已存在的任务处理器: {task_type}")

        self._handlers[task_type] = handler
        self._task_configs[task_type] = config
        logger.info(f"注册任务处理器: {task_type.value}")

    def get_handler(self, task_type: TaskType) -> Optional[TaskHandler]:
        """获取任务处理器"""
        return self._handlers.get(task_type)

    def get_config(self, task_type: TaskType) -> Optional[TaskConfig]:
        """获取任务配置"""
        return self._task_configs.get(task_type)

    def list_registered_tasks(self) -> List[TaskType]:
        """列出所有注册的任务类型"""
        return list(self._handlers.keys())

    def create_celery_tasks(self, celery_app: Celery) -> None:
        """为所有注册的处理器创建 Celery 任务"""
        for task_type, handler in self._handlers.items():
            config = self._task_configs[task_type]
            self._create_single_task(celery_app, task_type, handler, config)

    def _create_single_task(self, celery_app: Celery, task_type: TaskType,
                            handler: TaskHandler, config: TaskConfig) -> None:
        """创建单个 Celery 任务"""

        def task_wrapper(**kwargs):
            """任务包装器 - 添加通用逻辑"""
            try:
                # 验证配置
                if not handler.validate_config(kwargs):
                    return {
                        "status": "FAILED",
                        "error": "Invalid task configuration",
                        "task_type": task_type.value
                    }

                # 执行任务
                result = handler.execute(**kwargs)

                # 添加元数据
                result.update({
                    "task_type": task_type.value,
                    "timestamp": __import__("time").time()
                })

                return result

            except Exception as e:
                logger.error(f"任务执行失败 {task_type.value}: {str(e)}")
                return {
                    "status": "FAILED",
                    "error": str(e),
                    "task_type": task_type.value
                }

        # 设置任务属性
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

        # 注册到 Celery
        celery_app.task(**task_options)(task_wrapper)
        logger.info(f"创建 Celery 任务: {config.name}")


# 全局任务工厂实例
task_factory = TaskFactory()