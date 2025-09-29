"""
Download task repository - simplified wrapper around BaseRepository.
"""
from typing import List, Optional, Tuple, Dict
from ..models.download import DownloadTask, TaskStatus
from .base import BaseRepository


class DownloadTaskRepository(BaseRepository[DownloadTask]):
    """Download task repository providing CRUD operations."""
    _model = DownloadTask
    # ==================== Insert operations ====================

    @classmethod
    def create_task(cls,exchange: str,symbol: str,year: int,month: int,file_name: str = None,**kwargs) -> DownloadTask:
        """Create a download task."""
        if not file_name:
            file_name = f"{symbol}-aggTrades-{year}-{month:02d}.zip"

        return cls.create(
            exchange=exchange,
            symbol=symbol,
            year=year,
            month=month,
            file_name=file_name,
            **kwargs
        )

    @classmethod
    def create_batch_tasks(cls,exchange: str,symbols: List[str],year_month_pairs: List[Tuple[int, int]],**kwargs) \
            -> Tuple[List[DownloadTask], List[Dict]]:
        """Bulk create download tasks.
        Args:
            exchange: exchange name
            symbols: list of trading pairs
            year_month_pairs: list of (year, month)
            priority: task priority
            **kwargs: extra attributes

        Returns:
            Tuple[List[DownloadTask], List[Dict]]: (created tasks, skipped tasks info)
        """
        created_tasks = []
        skipped_tasks = []
        for symbol in symbols:
            for year, month in year_month_pairs:
                # Skip if the task already exists
                if cls.task_exists(exchange, symbol, year, month):
                    skipped_tasks.append({
                        'exchange': exchange,
                        'symbol': symbol,
                        'year': year,
                        'month': month,
                        'reason': 'task exsits'
                    })
                    continue
                try:
                    file_name = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
                    task = cls.create(
                        exchange=exchange,
                        symbol=symbol,
                        year=year,
                        month=month,
                        file_name=file_name,
                        status=TaskStatus.PENDING,
                        **kwargs
                    )
                    created_tasks.append(task)
                except Exception as e:
                    skipped_tasks.append({
                        'exchange': exchange,
                        'symbol': symbol,
                        'year': year,
                        'month': month,
                        'reason': f'Creation failed: {str(e)}'
                    })

        return created_tasks, skipped_tasks

    # ==================== Update operations ====================

    @classmethod
    def update_status(cls, task_id: int, status: TaskStatus) -> Optional[DownloadTask]:
        """Update task status."""
        return cls.update(task_id, status=status)

    @classmethod
    def update_file_info(
        cls,
        task_id: int,
        local_path: str = None,
        file_size: int = None
    ) -> Optional[DownloadTask]:
        """Update file metadata."""
        update_data = {}
        if local_path:
            update_data['local_path'] = local_path
        if file_size:
            update_data['file_size'] = file_size

        return cls.update(task_id, **update_data) if update_data else None

    # ==================== Delete operations ====================

    @classmethod
    def delete_task(cls, task_id: int) -> bool:
        """Delete task by id."""
        return cls.delete(task_id)

    @classmethod
    def delete_by_status(cls, status: TaskStatus) -> int:
        """Delete tasks by status."""
        return cls.delete_by_kwargs(status=status)

    # ==================== Query operations ====================

    @classmethod
    def get_all_tasks(cls,exchange: str = None,status: TaskStatus = None)-> List[DownloadTask]:
        """Retrieve tasks with optional filters."""
        filters = {}
        if exchange:
            filters['exchange'] = exchange
        if status:
            filters['status'] = status

        return cls.get_all(order_by="created_at",desc_order=True,**filters)



    @classmethod
    def task_exists(cls, exchange: str, symbol: str, year: int, month: int) -> bool:
        """Check if a task exists."""
        return cls.exists(
            exchange=exchange,
            symbol=symbol,
            year=year,
            month=month
        )

    @classmethod
    def get_task_id(cls,exchange:str,symbol:str,year:int,month:int) -> Optional[int]:
        return cls.get_by_kwargs(exchange=exchange,symbol=symbol,year=year,month=month).id