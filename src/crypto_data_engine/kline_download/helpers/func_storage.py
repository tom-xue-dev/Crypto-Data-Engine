from time import sleep
from helpers.config import Config
from helpers.model import DataFlag, SubTask


def make_sub_task(since, until, thread_count, fetch_data_func, parse_func=None):
    online_worker_count = thread_count
    local_worker_count = max(1, thread_count // Config("LOCAL_THREADS_RATIO"))
    test_since = since
    attempt_times = Config("MAX_ATTEMPT_TIMES")
    while attempt_times:
        test_raw_data, flag = fetch_data_func(since=test_since)
        test_data = parse_func(test_raw_data) if callable(parse_func) else test_raw_data
        if flag == DataFlag.NORMAL and len(test_data) > 0:
            delta = abs(test_data[0][0] - test_data[-1][0])
            since = test_data[0][0]
            timestamp_list = list(range(since, until, delta))
            return SubTask(
                timestamp_list,
                online_worker_count,
                local_worker_count,
            )
        else:
            attempt_times -= 1
            sleep(1)
    raise RuntimeError("生成子任务失败，尝试次数耗尽")


def parse_ohlcv(data):
    return list(map(lambda x: x[:5], data))


def parse_funding_rate(data):
    return [
        [item["info"]["fundingTime"], float(item["info"]["fundingRate"])]
        for item in data
    ]
