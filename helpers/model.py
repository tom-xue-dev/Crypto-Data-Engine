from dataclasses import dataclass
from enum import Enum


class DataFlag(Enum):
    NORMAL = 0
    ERROR = 1


@dataclass
class BasicInfo:
    '''
    基础信息，仅为数据文件夹创建使用
    '''
    exchange: str
    type: str
    symbol: str
    label: str

    def __str__(self):
        return (
            f"{self.exchange}_{self.type}_{self.symbol.replace('/','-')}_{self.label}"
        )


@dataclass
class SubTask:
    timestamp_list: list
    online_worker_count: int
    local_worker_count: int
