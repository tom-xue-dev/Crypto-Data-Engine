from pydantic import BaseModel


class DLReq(BaseModel):
    symbol: str
    start: str  # '2024-01-01'
    end: str
    interval: str = "1m"
    io_limit: int = 8      # 并发下载槽位
