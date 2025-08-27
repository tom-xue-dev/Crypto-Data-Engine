from typing import Optional, List

from pydantic import BaseModel, Field, validator


class DownloadRequest(BaseModel):
    """下载请求模型"""
    exchange: str = Field(..., description="交易所名称")
    symbol: str = Field(..., description="交易对")
    year: int = Field(..., ge=2010, le=2030, description="年份")
    month: int = Field(..., ge=1, le=12, description="月份")
    priority: Optional[int] = Field(0, description="优先级")
    file_url: Optional[str] = Field(None, description="文件下载链接")

    @validator('exchange')
    def validate_exchange(cls, v):
        supported_exchanges = ['binance', 'okx', 'bybit', 'huobi']
        if v.lower() not in supported_exchanges:
            raise ValueError(f'不支持的交易所: {v}')
        return v.lower()


class BatchDownloadRequest(BaseModel):
    """批量下载请求"""
    exchange: str = Field(..., description="交易所名称")
    symbols: List[str] = Field(..., min_items=1, description="交易对列表")
    year: int = Field(..., ge=2010, le=2030, description="年份")
    months: List[int] = Field(..., min_items=1, description="月份列表")

