"""
下载任务Repository - 基于BaseRepository的简化版本
"""
from typing import List, Optional, Tuple, Dict
from ..models.download import DownloadTask, TaskStatus
from .base import BaseRepository


class DownloadTaskRepository(BaseRepository[DownloadTask]):
    """下载任务Repository - 提供基本的CRUD操作"""
    _model = DownloadTask
    # ==================== 插入新记录 ====================

    @classmethod
    def create_task(cls,exchange: str,symbol: str,year: int,month: int,file_name: str = None,**kwargs) -> DownloadTask:
        """创建下载任务"""
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
        """
        批量创建下载任务
        Args:
            exchange: 交易所名称
            symbols: 交易对列表
            year_month_pairs: 年月对列表 [(year, month), ...]
            priority: 优先级
            **kwargs: 其他任务属性

        Returns:
            Tuple[List[DownloadTask], List[Dict]]: (创建的任务列表, 跳过的任务信息列表)
        """
        created_tasks = []
        skipped_tasks = []
        for symbol in symbols:
            for year, month in year_month_pairs:
                # 检查任务是否已存在
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
                        'reason': f'创建失败: {str(e)}'
                    })

        return created_tasks, skipped_tasks

    # ==================== 修改记录的某个字段 ====================

    @classmethod
    def update_status(cls, task_id: int, status: TaskStatus) -> Optional[DownloadTask]:
        """更新任务状态"""
        return cls.update(task_id, status=status)

    @classmethod
    def update_file_info(
        cls,
        task_id: int,
        local_path: str = None,
        file_size: int = None
    ) -> Optional[DownloadTask]:
        """更新文件信息"""
        update_data = {}
        if local_path:
            update_data['local_path'] = local_path
        if file_size:
            update_data['file_size'] = file_size

        return cls.update(task_id, **update_data) if update_data else None

    # ==================== 删除某个记录 ====================

    @classmethod
    def delete_task(cls, task_id: int) -> bool:
        """删除任务"""
        return cls.delete(task_id)

    @classmethod
    def delete_by_status(cls, status: TaskStatus) -> int:
        """根据状态删除任务"""
        return cls.delete_by_kwargs(status=status)

    # ==================== 查询当前所有记录等 ====================

    @classmethod
    def get_all_tasks(cls,exchange: str = None,status: TaskStatus = None)-> List[DownloadTask]:
        """获取所有任务"""
        filters = {}
        if exchange:
            filters['exchange'] = exchange
        if status:
            filters['status'] = status

        return cls.get_all(order_by="created_at",desc_order=True,**filters)



    @classmethod
    def task_exists(cls, exchange: str, symbol: str, year: int, month: int) -> bool:
        """检查任务是否存在"""
        return cls.exists(
            exchange=exchange,
            symbol=symbol,
            year=year,
            month=month
        )

    @classmethod
    def get_task_id(cls,exchange:str,symbol:str,year:int,month:int) -> Optional[int]:
        return cls.get_by_kwargs(exchange=exchange,symbol=symbol,year=year,month=month).id