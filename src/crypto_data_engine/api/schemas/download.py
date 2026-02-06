"""
Request schemas for download and aggregation endpoints.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DownloadRequest(BaseModel):
    """Download request schema."""
    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading pair")
    year: int = Field(..., ge=2010, le=2030, description="Year")
    month: int = Field(..., ge=1, le=12, description="Month")
    priority: Optional[int] = Field(0, description="Priority")
    file_url: Optional[str] = Field(None, description="Explicit download URL")

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, value: str) -> str:
        supported_exchanges = ["binance", "okx", "bybit", "huobi"]
        if value.lower() not in supported_exchanges:
            raise ValueError(f"unsupported exchange: {value}")
        return value.lower()


class BatchDownloadRequest(BaseModel):
    """Batch download request."""
    exchange: str = Field(..., description="Exchange name")
    symbols: List[str] = Field(..., description="List of trading pairs")
    year: int = Field(..., ge=2010, le=2030, description="Year")
    months: List[int] = Field(..., description="List of months")


BarType = Literal["time", "tick", "volume", "dollar", "imbalance"]


class AggregateRequest(BaseModel):
    """Aggregation request schema."""
    exchange: str = Field(..., description="Exchange name, e.g. binance")
    symbols: Optional[List[str]] = Field(
        None, description="Explicit trading pair list; falls back to DB results"
    )
    bar_type: str = Field(
        "volume_bar", description="Bar type, e.g. tick/volume/dollar or *_bar format"
    )
    threshold: Optional[int] = Field(
        None, description="Threshold for volume/dollar/tick bars"
    )
