from time import sleep, time

import ccxt
from helpers.config import Config
from helpers.model import DataFlag, BasicInfo, SubTask, Task
from helpers.utils import datetime_to_timestamp
from functools import partial


def _make_fetch_ohlcv_task(since_time, until_time, thread_count, fetch_data_func):
    online_worker_count = thread_count if thread_count else Config("GLOABLE_THREADS")
    local_worker_count = max(1, online_worker_count // Config("LOCAL_THREADS_RATIO"))
    test_since_timestamp = (
        datetime_to_timestamp(since_time)
        if since_time
        else datetime_to_timestamp(Config("DEFAULT_SINCE_TIME"))
    )
    util_timestamp = (
        datetime_to_timestamp(until_time) if until_time else int(time() * 1000)
    )
    attempt_times = Config("MAX_ATTEMPT_TIMES")
    while attempt_times:
        test_data, flag = fetch_data_func(test_since_timestamp)
        if flag == DataFlag.NORMAL and len(test_data) > 0:
            delta_timestamp = abs(test_data[0][0] - test_data[-1][0])
            since_timestamp = test_data[0][0]
            timestamp_list = list(
                range(
                    since_timestamp,
                    util_timestamp,
                    delta_timestamp,
                )
            )
            return SubTask(
                timestamp_list,
                online_worker_count,
                local_worker_count,
            )
        else:
            attempt_times -= 1
            sleep(1)
    raise RuntimeError("生成子任务失败，尝试次数耗尽")


class TaskAnalyzer:

    def __init__(self, task: Task, fetch_data_func=None):
        self.exchange_obj = self._initialize_ccxt_exchange(task.info)
        self.exchange_obj.enableRateLimit = (
            task.process_params.get("rate_limit")
            if task.process_params.get("rate_limit") is not None
            else Config("GLOBAL_RATE_LIMIT")
        )
        if task.name == "fetch_all_ohlcv":
            # 返回SubTask
            self._init_func = partial(
                _make_fetch_ohlcv_task,
                since_time=(
                    task.process_params.get("since_time")
                    if task.process_params
                    else None
                ),
                until_time=(
                    task.process_params.get("until_time")
                    if task.process_params
                    else None
                ),
                thread_count=(
                    task.process_params.get("thread_count")
                    if task.process_params
                    else None
                ),
                fetch_data_func=fetch_data_func,
            )
            # 需要since参数，返回data
            self._fetch_func = partial(
                self.exchange_obj.fetch_ohlcv,
                symbol=task.info.symbol,
                timeframe=task.info.interval,
            )
            task.save_params["columns"] = [
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

    def _initialize_ccxt_exchange(self, info: BasicInfo) -> ccxt.Exchange:
        retval: ccxt.Exchange = None
        if info.type == "spot":
            retval = getattr(ccxt, info.exchange)()
        elif info.type == "futures":
            retval = getattr(ccxt, info.exchange)(
                {"options": {"defaultType": "future"}}
            )
        else:
            raise ValueError(f"不支持的类型: {info.type}")
        return retval

    def fetch_data(self, **kwargs):
        return self._fetch_func(**kwargs)

    def initialize_sub_task(self, **kwargs):
        return self._init_func(**kwargs)

    def get_ccxt_exchange(self) -> ccxt.Exchange:
        return self.exchange_obj
