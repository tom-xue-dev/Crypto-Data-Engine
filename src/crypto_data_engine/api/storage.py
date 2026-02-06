"""
Shared task storage for backtest API routes.

Eliminates circular imports between backtest and visualization routers
by centralizing the in-memory task store.
"""
from typing import Any, Dict

# In-memory backtest task storage
# Key: task_id (str), Value: task dict with status, result, etc.
backtest_tasks: Dict[str, Dict[str, Any]] = {}


def get_task(task_id: str) -> Dict[str, Any] | None:
    """Get a backtest task by ID."""
    return backtest_tasks.get(task_id)


def set_task(task_id: str, task_data: Dict[str, Any]) -> None:
    """Create or update a backtest task."""
    backtest_tasks[task_id] = task_data


def delete_task(task_id: str) -> bool:
    """Delete a backtest task. Returns True if existed."""
    return backtest_tasks.pop(task_id, None) is not None


def list_tasks() -> Dict[str, Dict[str, Any]]:
    """Return all backtest tasks."""
    return backtest_tasks
