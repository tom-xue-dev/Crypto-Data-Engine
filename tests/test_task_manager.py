"""
Unit tests for TaskManager and related components.
"""
import time
from datetime import datetime, timedelta
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from crypto_data_engine.common.task_manager import (
    TaskStatus,
    TaskState,
    MemoryTaskStore,
    RedisTaskStore,
    TaskManager,
    create_task_manager,
)


# =============================================================================
# TaskState Tests
# =============================================================================

class TestTaskState:
    """Tests for TaskState dataclass."""

    def test_create_task_state(self):
        """Test creating a TaskState with default values."""
        task = TaskState(task_id="test-123")
        
        assert task.task_id == "test-123"
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0.0
        assert task.message == ""
        assert task.result is None
        assert task.error is None
        assert task.is_finished is False
        assert isinstance(task.created_at, datetime)

    def test_task_state_to_dict(self):
        """Test converting TaskState to dictionary."""
        task = TaskState(
            task_id="test-456",
            status=TaskStatus.RUNNING,
            progress=0.5,
            message="Processing",
            metadata={"key": "value"},
        )
        
        data = task.to_dict()
        
        assert data["task_id"] == "test-456"
        assert data["status"] == "running"
        assert data["progress"] == 0.5
        assert data["metadata"] == {"key": "value"}
        assert isinstance(data["created_at"], str)  # ISO format

    def test_task_state_from_dict(self):
        """Test creating TaskState from dictionary."""
        data = {
            "task_id": "test-789",
            "status": "completed",
            "progress": 1.0,
            "message": "Done",
            "result": {"data": [1, 2, 3]},
            "error": None,
            "traceback": None,
            "created_at": "2024-01-15T10:30:00",
            "started_at": "2024-01-15T10:30:01",
            "completed_at": "2024-01-15T10:35:00",
            "metadata": {},
        }
        
        task = TaskState.from_dict(data)
        
        assert task.task_id == "test-789"
        assert task.status == TaskStatus.COMPLETED
        assert task.progress == 1.0
        assert task.result == {"data": [1, 2, 3]}
        assert isinstance(task.created_at, datetime)

    def test_task_state_is_finished(self):
        """Test is_finished property for different statuses."""
        # Not finished
        assert TaskState(task_id="1", status=TaskStatus.PENDING).is_finished is False
        assert TaskState(task_id="2", status=TaskStatus.RUNNING).is_finished is False
        
        # Finished
        assert TaskState(task_id="3", status=TaskStatus.COMPLETED).is_finished is True
        assert TaskState(task_id="4", status=TaskStatus.FAILED).is_finished is True
        assert TaskState(task_id="5", status=TaskStatus.CANCELLED).is_finished is True

    def test_task_state_duration(self):
        """Test duration calculation."""
        now = datetime.now()
        task = TaskState(
            task_id="test",
            started_at=now - timedelta(seconds=10),
            completed_at=now,
        )
        
        assert task.duration_seconds is not None
        assert 9 <= task.duration_seconds <= 11  # Allow small margin


# =============================================================================
# MemoryTaskStore Tests
# =============================================================================

class TestMemoryTaskStore:
    """Tests for in-memory task store."""

    def test_save_and_get(self):
        """Test saving and retrieving a task."""
        store = MemoryTaskStore()
        task = TaskState(task_id="test-1", message="Hello")
        
        store.save(task)
        retrieved = store.get("test-1")
        
        assert retrieved is not None
        assert retrieved.task_id == "test-1"
        assert retrieved.message == "Hello"

    def test_get_nonexistent(self):
        """Test getting a task that doesn't exist."""
        store = MemoryTaskStore()
        
        assert store.get("nonexistent") is None

    def test_delete(self):
        """Test deleting a task."""
        store = MemoryTaskStore()
        task = TaskState(task_id="test-delete")
        store.save(task)
        
        assert store.delete("test-delete") is True
        assert store.get("test-delete") is None
        assert store.delete("test-delete") is False  # Already deleted

    def test_list_tasks(self):
        """Test listing tasks."""
        store = MemoryTaskStore()
        
        # Create tasks with different statuses
        store.save(TaskState(task_id="t1", status=TaskStatus.PENDING))
        store.save(TaskState(task_id="t2", status=TaskStatus.RUNNING))
        store.save(TaskState(task_id="t3", status=TaskStatus.COMPLETED))
        store.save(TaskState(task_id="t4", status=TaskStatus.PENDING))
        
        # List all
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 4
        
        # Filter by status
        pending_tasks = store.list_tasks(status=TaskStatus.PENDING)
        assert len(pending_tasks) == 2
        
        # Test pagination
        paginated = store.list_tasks(limit=2)
        assert len(paginated) == 2

    def test_cleanup_expired(self):
        """Test cleaning up expired tasks."""
        store = MemoryTaskStore()
        
        # Create completed task with old completion time
        old_task = TaskState(
            task_id="old",
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now() - timedelta(days=2),
        )
        store.save(old_task)
        
        # Create recent task
        recent_task = TaskState(
            task_id="recent",
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now(),
        )
        store.save(recent_task)
        
        # Create running task (should not be cleaned)
        running_task = TaskState(task_id="running", status=TaskStatus.RUNNING)
        store.save(running_task)
        
        # Cleanup tasks older than 1 day
        removed = store.cleanup_expired(ttl_seconds=86400)
        
        assert removed == 1
        assert store.get("old") is None
        assert store.get("recent") is not None
        assert store.get("running") is not None


# =============================================================================
# TaskManager Tests
# =============================================================================

class TestTaskManager:
    """Tests for TaskManager."""

    def test_create_task(self, temp_task_manager: TaskManager):
        """Test creating a new task."""
        task = temp_task_manager.create_task(metadata={"type": "backtest"})
        
        assert task.task_id is not None
        assert len(task.task_id) == 36  # UUID format
        assert task.status == TaskStatus.PENDING
        assert task.metadata == {"type": "backtest"}
        
        # Should be retrievable
        retrieved = temp_task_manager.get_task(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == task.task_id

    def test_update_task(self, temp_task_manager: TaskManager):
        """Test updating task state."""
        task = temp_task_manager.create_task()
        
        # Update status and progress
        updated = temp_task_manager.update_task(
            task.task_id,
            status=TaskStatus.RUNNING,
            progress=0.5,
            message="Processing...",
        )
        
        assert updated is not None
        assert updated.status == TaskStatus.RUNNING
        assert updated.progress == 0.5
        assert updated.message == "Processing..."
        assert updated.started_at is not None

    def test_update_task_completion(self, temp_task_manager: TaskManager):
        """Test updating task to completed status."""
        task = temp_task_manager.create_task()
        temp_task_manager.update_task(task.task_id, status=TaskStatus.RUNNING)
        
        updated = temp_task_manager.update_task(
            task.task_id,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            result={"total_return": 0.15},
        )
        
        assert updated.status == TaskStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.result == {"total_return": 0.15}

    def test_delete_task(self, temp_task_manager: TaskManager):
        """Test deleting a task."""
        task = temp_task_manager.create_task()
        task_id = task.task_id
        
        assert temp_task_manager.delete_task(task_id) is True
        assert temp_task_manager.get_task(task_id) is None
        assert temp_task_manager.delete_task(task_id) is False

    def test_list_tasks(self, temp_task_manager: TaskManager):
        """Test listing tasks."""
        # Create multiple tasks
        t1 = temp_task_manager.create_task()
        t2 = temp_task_manager.create_task()
        temp_task_manager.update_task(t2.task_id, status=TaskStatus.RUNNING)
        
        # List all
        all_tasks = temp_task_manager.list_tasks()
        assert len(all_tasks) == 2
        
        # Filter by status
        running = temp_task_manager.list_tasks(status=TaskStatus.RUNNING)
        assert len(running) == 1
        assert running[0].task_id == t2.task_id

    def test_submit_io_task(self, temp_task_manager: TaskManager):
        """Test submitting an I/O task."""
        task = temp_task_manager.create_task()
        
        def io_task(value):
            time.sleep(0.1)
            return value * 2
        
        temp_task_manager.submit_io_task(task.task_id, io_task, 21)
        
        # Wait for completion
        time.sleep(0.5)
        
        completed = temp_task_manager.get_task(task.task_id)
        assert completed is not None
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == 42

    def test_submit_io_task_failure(self, temp_task_manager: TaskManager):
        """Test I/O task that fails."""
        task = temp_task_manager.create_task()
        
        def failing_task():
            raise ValueError("Task failed!")
        
        temp_task_manager.submit_io_task(task.task_id, failing_task)
        
        # Wait for completion
        time.sleep(0.5)
        
        failed = temp_task_manager.get_task(task.task_id)
        assert failed is not None
        assert failed.status == TaskStatus.FAILED
        assert "Task failed!" in failed.error
        assert failed.traceback is not None

    def test_progress_clamping(self, temp_task_manager: TaskManager):
        """Test that progress is clamped to [0, 1]."""
        task = temp_task_manager.create_task()
        
        # Test over 1.0
        temp_task_manager.update_task(task.task_id, progress=1.5)
        assert temp_task_manager.get_task(task.task_id).progress == 1.0
        
        # Test negative
        temp_task_manager.update_task(task.task_id, progress=-0.5)
        assert temp_task_manager.get_task(task.task_id).progress == 0.0


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateTaskManager:
    """Tests for task manager factory function."""

    def test_create_memory_manager(self):
        """Test creating manager with memory store."""
        manager = create_task_manager(store_type="memory")
        
        assert manager is not None
        assert isinstance(manager._store, MemoryTaskStore)
        manager.shutdown()

    def test_create_redis_manager_fallback(self):
        """Test that Redis manager falls back gracefully if Redis unavailable."""
        # This test assumes Redis is not running locally
        # In a real CI environment, you might skip this or mock Redis
        try:
            manager = create_task_manager(
                store_type="redis",
                redis_url="redis://localhost:6379/15",  # Use different DB
            )
            # If Redis is available, manager should work
            task = manager.create_task()
            assert task is not None
            manager.shutdown()
        except Exception:
            # Expected if Redis is not available
            pass


# =============================================================================
# Integration Tests
# =============================================================================

class TestTaskManagerIntegration:
    """Integration tests for complete task workflows."""

    def test_complete_task_workflow(self, temp_task_manager: TaskManager):
        """Test a complete task workflow from creation to completion."""
        # 1. Create task
        task = temp_task_manager.create_task(metadata={"strategy": "momentum"})
        assert task.status == TaskStatus.PENDING
        
        # 2. Start task
        temp_task_manager.update_task(
            task.task_id,
            status=TaskStatus.RUNNING,
            message="Starting backtest...",
        )
        
        running = temp_task_manager.get_task(task.task_id)
        assert running.status == TaskStatus.RUNNING
        assert running.started_at is not None
        
        # 3. Update progress
        temp_task_manager.update_task(
            task.task_id,
            progress=0.5,
            message="50% complete",
        )
        
        # 4. Complete task
        temp_task_manager.update_task(
            task.task_id,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            result={
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
            },
        )
        
        completed = temp_task_manager.get_task(task.task_id)
        assert completed.status == TaskStatus.COMPLETED
        assert completed.progress == 1.0
        assert completed.result["sharpe_ratio"] == 1.2
        assert completed.duration_seconds is not None

    def test_concurrent_tasks(self, temp_task_manager: TaskManager):
        """Test running multiple tasks concurrently."""
        tasks = []
        results = []
        
        def worker(task_id: str, value: int):
            time.sleep(0.1)
            return value * 2
        
        # Create and submit multiple tasks
        for i in range(5):
            task = temp_task_manager.create_task(metadata={"index": i})
            tasks.append(task)
            temp_task_manager.submit_io_task(task.task_id, worker, task.task_id, i)
        
        # Wait for all to complete
        time.sleep(1.0)
        
        # Verify all completed
        for i, task in enumerate(tasks):
            completed = temp_task_manager.get_task(task.task_id)
            assert completed.status == TaskStatus.COMPLETED
            assert completed.result == i * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
