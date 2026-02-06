"""
Pytest configuration and shared fixtures.
"""
import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Get data directory."""
    return project_root / "data"


@pytest.fixture
def temp_task_manager():
    """Create a temporary in-memory task manager for testing."""
    from crypto_data_engine.common.task_manager import TaskManager, MemoryTaskStore
    
    manager = TaskManager(store=MemoryTaskStore())
    yield manager
    manager.shutdown(wait=True)
