import datetime
from pydantic import BaseModel
from pydantic.v1 import Field


class TickDownloadRequest(BaseModel):
    symbol: str = Field(..., example="BTCUSDT")
    start_time: str = Field(..., example="BTCUSDT")
    end_time: str = Field(..., example="BTCUSDT")

class CryptoSymbolRequest(BaseModel):
    suffix: str = Field(..., example="USDT")