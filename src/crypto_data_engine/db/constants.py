from enum import Enum as PyEnum
class TaskStatus(PyEnum):
    """Task status enumeration."""
    PENDING = "pending"  # Not started
    DOWNLOADING = "downloading"  # In progress
    DOWNLOADED = "downloaded"  # Download finished
    EXTRACTING = "extracting"  # Extracting data
    EXTRACTED = "extracted"  # Extraction finished

    PROCESSING = "processing"  # Processing/converting
    COMPLETED = "completed"  # Fully completed
    FAILED = "failed"  # Failed
    SKIPPED = "skipped"  # Skipped
    RETRY = "retry"  # Waiting for retry