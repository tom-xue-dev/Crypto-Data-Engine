"""
Download service database models.
"""
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Boolean,
    Enum, Index, UniqueConstraint, BigInteger
)
from crypto_data_engine.db.constants import TaskStatus
from crypto_data_engine.db.models.Base import Base


class DownloadTask(Base):
    """Download task table."""
    __tablename__ = "download_tasks"
    id = Column(BigInteger, primary_key=True, index=True)
    exchange = Column(String(50), nullable=False, comment="Exchange name (binance, okx, etc.)")
    symbol = Column(String(50), nullable=False, comment="Trading pair (BTCUSDT, ETHUSDT, etc.)")
    year = Column(Integer, nullable=False, comment="Year")
    month = Column(Integer, nullable=False, comment="Month")
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING, comment="Task status")
    file_name = Column(String(255), comment="File name")
    file_size = Column(BigInteger, comment="File size (bytes)")
    local_path = Column(String(500), comment="Local file path")
    task_start = Column(DateTime, comment="Start time")
    task_end = Column(DateTime, comment="Complete time")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint('exchange', 'symbol', 'year', 'month', name='uq_download_task'),
        # Index('idx_download_status', 'status'),
        # Index('idx_download_exchange', 'exchange'),
        # Index('idx_download_symbol', 'symbol'),
        {'comment': 'Download task table'}
    )

    def __repr__(self):
        return f"<DownloadTask(id={self.id}, {self.exchange}/{self.symbol}/{self.year}-{self.month:02d}, status={self.status.value})>"

