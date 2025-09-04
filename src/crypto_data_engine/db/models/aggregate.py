from sqlalchemy import (
    Column, BigInteger, String, Integer, Enum, Date, DateTime, BigInteger as SA_BigInt,
    UniqueConstraint, Index
)

from crypto_data_engine.db.constants import TaskStatus
from crypto_data_engine.db.models.Base import Base


class AggregateTask(Base):
    """聚合任务表（bars/features 等产出）"""
    __tablename__ = "aggregate_tasks"

    id = Column(BigInteger, primary_key=True, index=True)
    exchange   = Column(String(50),  nullable=False, comment="交易所名称 (binance, okx, etc.)")
    symbol     = Column(String(50),  nullable=False, comment="交易对 (BTCUSDT, ETHUSDT, etc.)")
    bar_type   = Column(String(30),  nullable=False, comment="聚合类型: time/dollar/volume/tick/imbalance 等")
    # 分区与时间范围（用于快速定位与查找/去重）
    part_date  = Column(Date,        nullable=False, comment="数据分区日（按天产出常用）")
    # 状态与产物信息
    status      = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING, comment="任务状态")
    file_name = Column(String(255),  nullable=True, comment="输出文件名（可选）")
    file_path = Column(String(500),  nullable=True, comment="输出路径（本地/对象存储）")
    # 任务起止
    task_start  = Column(DateTime, nullable=True, comment="开始时间")
    task_end    = Column(DateTime, nullable=True, comment="完成时间")

    # 幂等键（可选）：避免重复提交完全相同的聚合任务
    idempotency_key = Column(String(200), nullable=True, comment="幂等键（可选）")

    __table_args__ = (
        # 一个 symbol 在某个聚合配置与分区日上只应产出一次（可多版本共存）
        UniqueConstraint('exchange', 'symbol', 'bar_type', 'interval', 'version', 'part_date',
                         name='uq_aggregate_task_unique_output'),
        Index('idx_agg_status', 'status'),
        Index('idx_agg_exchange_symbol', 'exchange', 'symbol'),
        Index('idx_agg_part_date', 'part_date'),
        {'comment': '聚合任务表'}
    )

    def __repr__(self):
        return (f"<AggregateTask(id={self.id}, {self.exchange}/{self.symbol}, "
                f"{self.bar_type}, "
                f"part={self.part_date}, status={self.status.value})>")
