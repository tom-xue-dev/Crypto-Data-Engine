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

class BarProcessorRequest(BaseModel):
    bar_type: BarType
    # 通用
    exchange_name: str = Field(..., description="binance/okx/...")
    symbol: str = Field(..., description="BTCUSDT 等")
    process_num_limit: int = 4
    raw_data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    suffix_filter: Optional[str] = None  # e.g. '.parquet'
    kwargs: Optional[Dict[str, Any]] = None # you need to provide suitable args for each bar_type

    @root_validator
    def validate_by_bar_type(cls, v):
        bt = v.get('bar_type')
        thr = v.get('threshold')
        itv = v.get('interval')
        ema = v.get('ema_window')

        if bt == 'time':
            if not itv:
                raise ValueError("time bar 需要 interval（如 '1s'/'1m'）")
        elif bt == 'tick':
            # tick bar 一般不需要 interval/thr；可选：固定 N 笔合一
            pass
        elif bt in ('volume', 'dollar'):
            if thr is None or thr <= 0:
                raise ValueError(f"{bt} bar 需要正整数 threshold")
        elif bt == 'imbalance':
            # 常见：以 tick/volume/dollar 不平衡触发；给个需要的窗口
            if ema is None or ema <= 0:
                raise ValueError("imbalance bar 需要 ema_window")
        return v
