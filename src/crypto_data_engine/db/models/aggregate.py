from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Integer,
    Enum,
    Date,
    DateTime,
    UniqueConstraint,
    Index,
)

from crypto_data_engine.db.constants import TaskStatus
from crypto_data_engine.db.models.Base import Base


class AggregateTask(Base):
    """Aggregation task table (bars/features outputs)."""

    __tablename__ = "aggregate_tasks"

    id = Column(BigInteger, primary_key=True, index=True)
    exchange = Column(String(50), nullable=False, comment="exchange name (binance, okx, etc.)")
    symbol = Column(String(50), nullable=False, comment="symbol (BTCUSDT, ETHUSDT, etc.)")
    bar_type = Column(String(30), nullable=False, comment="aggregation type: time/dollar/volume/tick/imbalance")
    # partition and time range (for locate/search/dedup)
    part_date = Column(Date, nullable=False, comment="partition date (e.g., per day if applicable)")
    # status and artifact info
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING, comment="task status")
    file_name = Column(String(255), nullable=True, comment="output file name (optional)")
    file_path = Column(String(500), nullable=True, comment="output path (local or object storage)")
    # timing
    task_start = Column(DateTime, nullable=True, comment="start time")
    task_end = Column(DateTime, nullable=True, comment="finish time")

    # idempotency (optional)
    idempotency_key = Column(String(200), nullable=True, comment="idempotency key (optional)")

    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "symbol",
            "bar_type",
            "part_date",
            name="uq_aggregate_task_unique_output",
        ),
        Index("idx_agg_status", "status"),
        Index("idx_agg_exchange_symbol", "exchange", "symbol"),
        Index("idx_agg_part_date", "part_date"),
        {"comment": "aggregation tasks"},
    )

    def __repr__(self):
        return (
            f"<AggregateTask(id={self.id}, {self.exchange}/{self.symbol}, "
            f"{self.bar_type}, part={self.part_date}, status={self.status.value})>"
        )

