from dataclasses import dataclass
from enum import Enum


class DataFlag(Enum):
    NORMAL = 0
    ERROR = 1


@dataclass
class BasicInfo:
    exchange: str
    type: str
    symbol: str
    interval: str

    def __str__(self):
        return f"{self.exchange}_{self.type}_{self.symbol.replace('/','-')}_{self.interval}"


@dataclass
class Task:
    """
    任务类：描述一个高级指令的所有信息

    属性:

    - info (`BasicInfo`): 
      基本信息：包括交易所名称，类型，交易对，间隔。

    - name (`str`): 
      当前支持的所有指令名称：
        - `fetch_all_ohlcv`

    - process_params (`dict`): 
      加工参数（均为选填）：
        - `since_time` (`datetime` or `str`): 开始时间，默认值存储在配置文件的 `DEFAULT_SINCE_TIME`。
        - `util_time` (`datetime` or `str`): 结束时间，默认值为当前时间。
        - `rate_limit` (`bool`): 是否开启速率限制，默认值为 `False`。
        - `thread_count` (`int`): 线程数，默认值存储在配置文件的 `GLOBAL_THREADS`。

    - save_params (`dict`): 
      保存参数（均为选填）：
        - `drop_last` (`bool`): 是否去除最后一个数据，默认值为 `True`。
        - `fix_integrity` (`bool`): 是否对缺失数据进行补全，默认值为 `True`。
        - `save_missing_times` (`bool`): 是否保存缺失数据到文本，默认值为 `True`。
    """

    info: BasicInfo
    name: str
    process_params: dict
    save_params: dict


@dataclass
class SubTask:
    timestamp_list: list
    online_worker_count: int
    local_worker_count: int
