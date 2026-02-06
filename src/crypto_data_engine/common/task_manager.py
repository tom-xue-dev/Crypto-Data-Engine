"""
Lightweight task manager for backtest system.

Replaces Celery with a simpler approach using:
- FastAPI BackgroundTasks for async execution
- Redis or in-memory storage for task state
- ProcessPoolExecutor for CPU-bound tasks
- ThreadPoolExecutor for I/O-bound tasks
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Task Status and State
# =============================================================================

class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    """Represents the state of a task."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format strings
        for key in ["created_at", "started_at", "completed_at"]:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        """Create TaskState from dictionary."""
        # Convert ISO strings back to datetime
        for key in ["created_at", "started_at", "completed_at"]:
            if data.get(key) is not None and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        # Convert status string to enum
        if isinstance(data.get("status"), str):
            data["status"] = TaskStatus(data["status"])
        return cls(**data)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def is_finished(self) -> bool:
        """Check if task has finished (completed, failed, or cancelled)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)


# =============================================================================
# Task Store Interface and Implementations
# =============================================================================

class TaskStore(ABC):
    """Abstract interface for task state storage."""

    @abstractmethod
    def save(self, task: TaskState) -> None:
        """Save task state."""
        pass

    @abstractmethod
    def get(self, task_id: str) -> Optional[TaskState]:
        """Get task state by ID."""
        pass

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """Delete task state."""
        pass

    @abstractmethod
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TaskState]:
        """List tasks with optional filtering."""
        pass

    @abstractmethod
    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove tasks older than TTL. Returns count of removed tasks."""
        pass


class MemoryTaskStore(TaskStore):
    """In-memory task store for development and testing."""

    def __init__(self):
        self._tasks: Dict[str, TaskState] = {}

    def save(self, task: TaskState) -> None:
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    def delete(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TaskState]:
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[offset : offset + limit]

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove tasks older than TTL."""
        now = datetime.now()
        expired_ids = [
            task_id
            for task_id, task in self._tasks.items()
            if task.is_finished
            and task.completed_at
            and (now - task.completed_at).total_seconds() > ttl_seconds
        ]
        for task_id in expired_ids:
            del self._tasks[task_id]
        return len(expired_ids)


class RedisTaskStore(TaskStore):
    """Redis-based task store for production."""

    TASK_KEY_PREFIX = "task:"
    TASK_LIST_KEY = "tasks:all"

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self._redis.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except ImportError:
            raise ImportError("redis package required. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _task_key(self, task_id: str) -> str:
        return f"{self.TASK_KEY_PREFIX}{task_id}"

    def save(self, task: TaskState) -> None:
        key = self._task_key(task.task_id)
        data = json.dumps(task.to_dict())
        self._redis.set(key, data)
        # Add to task list set with timestamp score
        self._redis.zadd(
            self.TASK_LIST_KEY,
            {task.task_id: task.created_at.timestamp()},
        )

    def get(self, task_id: str) -> Optional[TaskState]:
        key = self._task_key(task_id)
        data = self._redis.get(key)
        if data is None:
            return None
        return TaskState.from_dict(json.loads(data))

    def delete(self, task_id: str) -> bool:
        key = self._task_key(task_id)
        result = self._redis.delete(key)
        self._redis.zrem(self.TASK_LIST_KEY, task_id)
        return result > 0

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TaskState]:
        # Get task IDs from sorted set (newest first)
        task_ids = self._redis.zrevrange(
            self.TASK_LIST_KEY,
            offset,
            offset + limit - 1,
        )

        tasks = []
        for task_id in task_ids:
            task = self.get(task_id)
            if task is not None:
                if status is None or task.status == status:
                    tasks.append(task)

        return tasks

    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove tasks older than TTL."""
        cutoff_timestamp = time.time() - ttl_seconds
        # Get expired task IDs
        expired_ids = self._redis.zrangebyscore(
            self.TASK_LIST_KEY,
            "-inf",
            cutoff_timestamp,
        )

        count = 0
        for task_id in expired_ids:
            task = self.get(task_id)
            if task is not None and task.is_finished:
                self.delete(task_id)
                count += 1

        return count


# =============================================================================
# Task Manager
# =============================================================================

class TaskManager:
    """
    Lightweight task manager for executing and tracking tasks.

    Replaces Celery with simpler, more debuggable approach.
    """

    def __init__(
        self,
        store: Optional[TaskStore] = None,
        max_io_threads: int = 16,
        max_compute_processes: int = 8,
    ):
        """
        Initialize task manager.

        Args:
            store: Task state storage backend. Defaults to MemoryTaskStore.
            max_io_threads: Maximum threads for I/O operations.
            max_compute_processes: Maximum processes for CPU operations.
        """
        self._store = store or MemoryTaskStore()
        self._max_io_threads = max_io_threads
        self._max_compute_processes = max_compute_processes

        # Executors are created lazily
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None

        # Track running tasks
        self._running_futures: Dict[str, Any] = {}

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool for I/O tasks."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self._max_io_threads,
                thread_name_prefix="task_io_",
            )
        return self._thread_pool

    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool for CPU tasks."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self._max_compute_processes,
            )
        return self._process_pool

    def create_task(self, metadata: Optional[Dict[str, Any]] = None) -> TaskState:
        """Create a new task and return its state."""
        task_id = str(uuid.uuid4())
        task = TaskState(
            task_id=task_id,
            metadata=metadata or {},
        )
        self._store.save(task)
        logger.info(f"Created task {task_id}")
        return task

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task state by ID."""
        return self._store.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        traceback: Optional[str] = None,
    ) -> Optional[TaskState]:
        """Update task state."""
        task = self._store.get(task_id)
        if task is None:
            return None

        if status is not None:
            task.status = status
            if status == TaskStatus.RUNNING and task.started_at is None:
                task.started_at = datetime.now()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.completed_at = datetime.now()

        if progress is not None:
            task.progress = min(max(progress, 0.0), 1.0)
        if message is not None:
            task.message = message
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if traceback is not None:
            task.traceback = traceback

        self._store.save(task)
        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete task."""
        return self._store.delete(task_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TaskState]:
        """List tasks with optional filtering."""
        return self._store.list_tasks(status=status, limit=limit, offset=offset)

    def submit_io_task(
        self,
        task_id: str,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> None:
        """
        Submit an I/O-bound task for execution in thread pool.

        Args:
            task_id: Task ID to track
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
        """
        self.update_task(task_id, status=TaskStatus.RUNNING)

        def wrapper():
            try:
                result = func(*args, **kwargs)
                self.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=1.0,
                    result=result,
                )
                return result
            except Exception as e:
                import traceback as tb
                self.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    traceback=tb.format_exc(),
                )
                raise

        future = self.thread_pool.submit(wrapper)
        self._running_futures[task_id] = future

    def submit_compute_task(
        self,
        task_id: str,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> None:
        """
        Submit a CPU-bound task for execution in process pool.

        Note: The function and arguments must be picklable.

        Args:
            task_id: Task ID to track
            func: Function to execute (must be module-level, not lambda)
            *args, **kwargs: Arguments to pass to function
        """
        self.update_task(task_id, status=TaskStatus.RUNNING)

        # For process pool, we need to handle results differently
        # since the function runs in a separate process
        future = self.process_pool.submit(func, *args, **kwargs)
        self._running_futures[task_id] = future

        # Add callback to update task state when done
        def on_complete(fut):
            try:
                result = fut.result()
                self.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=1.0,
                    result=result,
                )
            except Exception as e:
                import traceback as tb
                self.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    traceback=tb.format_exc(),
                )

        future.add_done_callback(on_complete)

    def cancel_task(self, task_id: str) -> bool:
        """
        Attempt to cancel a running task.

        Returns True if cancellation was successful.
        """
        future = self._running_futures.get(task_id)
        if future is not None:
            cancelled = future.cancel()
            if cancelled:
                self.update_task(task_id, status=TaskStatus.CANCELLED)
                del self._running_futures[task_id]
                return True
        return False

    def cleanup_expired(self, ttl_seconds: int = 86400) -> int:
        """Remove expired tasks older than TTL."""
        return self._store.cleanup_expired(ttl_seconds)

    def shutdown(self, wait: bool = True):
        """Shutdown executor pools."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=wait)
            self._thread_pool = None
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=wait)
            self._process_pool = None
        logger.info("Task manager shutdown complete")


# =============================================================================
# Factory function
# =============================================================================

def create_task_manager(
    store_type: str = "memory",
    redis_url: str = "redis://localhost:6379/0",
    max_io_threads: int = 16,
    max_compute_processes: int = 8,
) -> TaskManager:
    """
    Create a task manager with the specified configuration.

    Args:
        store_type: "memory" or "redis"
        redis_url: Redis connection URL (only used if store_type is "redis")
        max_io_threads: Maximum threads for I/O operations
        max_compute_processes: Maximum processes for CPU operations

    Returns:
        Configured TaskManager instance
    """
    if store_type == "redis":
        store = RedisTaskStore(redis_url=redis_url)
    else:
        store = MemoryTaskStore()

    return TaskManager(
        store=store,
        max_io_threads=max_io_threads,
        max_compute_processes=max_compute_processes,
    )


def get_task_manager() -> TaskManager:
    """
    Get the global task manager instance.

    Creates one if it doesn't exist, using settings from config.
    """
    global _task_manager
    if _task_manager is None:
        from crypto_data_engine.common.config.config_settings import settings
        config = settings.task_cfg
        _task_manager = create_task_manager(
            store_type=config.task_store,
            redis_url=config.redis_url,
            max_io_threads=config.max_io_threads,
            max_compute_processes=config.max_compute_processes,
        )
    return _task_manager


# Global instance (lazy initialized)
_task_manager: Optional[TaskManager] = None
