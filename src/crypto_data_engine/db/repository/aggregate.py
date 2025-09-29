from typing import List, Tuple, Dict, Optional
import os



from crypto_data_engine.db.constants import TaskStatus
from crypto_data_engine.db.models.aggregate import AggregateTask
from crypto_data_engine.db.models.download import DownloadTask
from crypto_data_engine.db.repository.base import BaseRepository
from crypto_data_engine.db.session import with_db_session


class AggregateTaskRepository(BaseRepository[AggregateTask]):
    _model = AggregateTask

    @classmethod
    def create_task(cls,exchange: str,symbol: str,bar_type: str,
        file_name: Optional[str] = None,file_path: Optional[str] = None,
        idempotency_key: Optional[str] = None,**kwargs
    ) -> AggregateTask:
        return cls.create(
            exchange=exchange,
            symbol=symbol,
            bar_type=bar_type,
            status=TaskStatus.PENDING,
            idempotency_key=idempotency_key,
            file_name=file_name,
            file_path=file_path,
            **kwargs
        )

    @classmethod
    def create_batch_tasks(
        cls,
        exchange: str,symbols: List[str],bar_type: str,
        default_output_dir: Optional[str] = None,default_ext: str = ".parquet",
        **kwargs
    ) -> Tuple[List[AggregateTask], List[Dict]]:
        """
        Bulk create aggregation tasks (mirrors `create_task` parameters).
        - If `file_name` is absent, generate `{symbol}-{bar_type}{default_ext}`.
        - When `default_output_dir` is provided, auto-build `file_path`.
        - Existence check: exchange + symbol + bar_type + (file_name or file_path).
        Returns:
            (created_tasks, skipped_tasks)
        """
        created: List[AggregateTask] = []
        skipped: List[Dict] = []

        for symbol in symbols:
            # 1) Generate file_name / file_path when not explicitly provided
            file_name = kwargs.get("file_name")
            file_path = kwargs.get("file_path")

            if not file_name:
                file_name = f"{symbol}-{bar_type}{default_ext or ''}"

            if not file_path and default_output_dir:
                file_path = os.path.join(default_output_dir, file_name)

            # 2) Deduplicate: prefer file_path > file_name > tuple fallback
            if cls.task_exists(
                exchange=exchange,
                symbol=symbol,
                bar_type=bar_type,
                file_name=file_name,
                file_path=file_path
            ):
                skipped.append({
                    "exchange": exchange,
                    "symbol": symbol,
                    "bar_type": bar_type,
                    "file_name": file_name,
                    "file_path": file_path,
                    "reason": "task exists"
                })
                continue

            # 3) Create
            try:
                task = cls.create_task(
                    exchange=exchange,
                    symbol=symbol,
                    bar_type=bar_type,
                    file_name=file_name,
                    file_path=file_path,
                    **{k: v for k, v in kwargs.items() if k not in ("file_name", "file_path")}
                )
                created.append(task)
            except Exception as e:
                skipped.append({
                    "exchange": exchange,
                    "symbol": symbol,
                    "bar_type": bar_type,
                    "file_name": file_name,
                    "file_path": file_path,
                    "reason": f"Creation failed: {str(e)}"
                })

        return created, skipped


    @classmethod
    def get_all_tasks(cls,exchange: str = None,status: TaskStatus = None)-> List[AggregateTask]:
        """Retrieve all tasks."""
        filters = {}
        if exchange:
            filters['exchange'] = exchange
        if status:
            filters['status'] = status
        # Order by part_date if available; fallback to id
        return cls.get_all(order_by="part_date",desc_order=True,**filters)


    from sqlalchemy.orm import Session
    @classmethod
    @with_db_session
    def get_ready_pairs(cls, exchange_name: str, db: Session):
        """
        Download tasks finished and no aggregation tasks exist for (exchange, symbol)
        """
        A, D = AggregateTask, DownloadTask
        from sqlalchemy import and_
        subq_exists_agg = db.query(A.id).filter(
            and_(A.exchange == D.exchange, A.symbol == D.symbol)
        ).exists()
        rows = (
            db.query(D.exchange, D.symbol)
            .filter(
                D.exchange == exchange_name,
                D.status == TaskStatus.COMPLETED,
                ~subq_exists_agg                  # anti-join: NOT EXISTS
            )
            .distinct()
            .all()
        )
        return rows  # List[Tuple[str, str]]
    # ======= Helper: existence check (paired with create_batch_tasks) =======

    @classmethod
    def task_exists(
        cls,
        exchange: str,
        symbol: str,
        bar_type: str,
        file_name: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> bool:
        """
        Existence check precedence:
        1) With file_path → (exchange, symbol, bar_type, file_path)
        2) else with file_name → (exchange, symbol, bar_type, file_name)
        3) otherwise fall back to coarse (exchange, symbol, bar_type)
        """
        if file_path:
            return cls.exists(
                exchange=exchange, symbol=symbol, bar_type=bar_type, file_path=file_path
            )
        if file_name:
            return cls.exists(
                exchange=exchange, symbol=symbol, bar_type=bar_type, file_name=file_name
            )
        return cls.exists(exchange=exchange, symbol=symbol, bar_type=bar_type)
