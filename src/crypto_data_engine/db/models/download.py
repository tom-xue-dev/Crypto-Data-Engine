"""
下载服务数据库模型
"""
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Boolean,
    Enum, Index, UniqueConstraint, BigInteger
)
from crypto_data_engine.db.constants import TaskStatus
from crypto_data_engine.db.models.Base import Base


class DownloadTask(Base):
    """下载任务表"""
    __tablename__ = "download_tasks"
    id = Column(BigInteger, primary_key=True, index=True)
    exchange = Column(String(50), nullable=False, comment="交易所名称 (binance, okx, etc.)")
    symbol = Column(String(50), nullable=False, comment="交易对 (BTCUSDT, ETHUSDT, etc.)")
    year = Column(Integer, nullable=False, comment="年份")
    month = Column(Integer, nullable=False, comment="月份")
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING, comment="任务状态")
    file_name = Column(String(255), comment="文件名")
    file_size = Column(BigInteger, comment="文件大小(字节)")
    local_path = Column(String(500), comment="本地文件路径")
    task_start = Column(DateTime, comment="开始时间")
    task_end = Column(DateTime, comment="完成时间")

    # 唯一约束
    __table_args__ = (
        UniqueConstraint('exchange', 'symbol', 'year', 'month', name='uq_download_task'),
        # Index('idx_download_status', 'status'),
        # Index('idx_download_exchange', 'exchange'),
        # Index('idx_download_symbol', 'symbol'),
        {'comment': '下载任务表'}
    )

    def __repr__(self):
        return f"<DownloadTask(id={self.id}, {self.exchange}/{self.symbol}/{self.year}-{self.month:02d}, status={self.status.value})>"

