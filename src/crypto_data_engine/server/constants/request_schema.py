from typing import Optional, List
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, conint, root_validator, field_validator


class DownloadRequest(BaseModel):
    """下载请求模型"""
    exchange: str = Field(..., description="交易所名称")
    symbol: str = Field(..., description="交易对")
    year: int = Field(..., ge=2010, le=2030, description="年份")
    month: int = Field(..., ge=1, le=12, description="月份")
    priority: Optional[int] = Field(0, description="优先级")
    file_url: Optional[str] = Field(None, description="文件下载链接")

    @field_validator('exchange')
    def validate_exchange(cls, v):
        supported_exchanges = ['binance', 'okx', 'bybit', 'huobi']
        if v.lower() not in supported_exchanges:
            raise ValueError(f'unsupported exchange: {v}')
        return v.lower()


class BatchDownloadRequest(BaseModel):
    """批量下载请求"""
    exchange: str = Field(..., description="交易所名称")
    symbols: List[str] = Field(..., description="交易对列表")
    year: int = Field(..., ge=2010, le=2030, description="年份")
    months: List[int] = Field(..., description="月份列表")




BarType = Literal['time', 'tick', 'volume', 'dollar', 'imbalance']

class AggregateRequest(BaseModel):
    exchange: str = Field(..., description="交易所，如 binance")
    symbols: Optional[List[str]] = Field(None, description="指定交易对；为空则自动从DB筛选")
    bar_type: str = Field("volume_bar", description="tick/volume/dollar 或 *_bar 形式")
    threshold: Optional[int] = Field(None, description="阈值（对 volume/dollar/tick 有效）")
