import datetime

import ccxt
import pytz

from fake_useragent import UserAgent

from helpers.model import BasicInfo


_DELTA_TIMESTAMP_MAPPING = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "M": 2628000,
}


def timeframe_to_timestamp(timeframe, rate=1000):
    return (
        _DELTA_TIMESTAMP_MAPPING[timeframe[-1]]
        * int(timeframe[:-1])
        * rate
    )


def timestamp_to_datetime(timestamp, rate=1000):
    """
    将时间戳（毫秒）转换为系统时间
    :param timestamp: 时间戳（毫秒）
    :return: 转换后的系统时间字符串
    """
    timestamp = int(int(timestamp) / rate)
    dt_object = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")  # 格式化为“年-月-日 时:分:秒”


def datetime_to_timestamp(dt: str | datetime.datetime, rate=1000):
    """
    将格式化后的 datetime 字符串转换为时间戳（毫秒）
    :param datetime_str: 格式化的 datetime 字符串，例如 '2024-11-15 14:30:00'
    :return: 对应的时间戳（毫秒）
    """
    datetime_format = "%Y-%m-%d %H:%M:%S"
    utc_tz = pytz.timezone("UTC")
    dt_object = (
        datetime.datetime.strptime(dt, datetime_format) if isinstance(dt, str) else dt
    )
    dt_object = utc_tz.localize(dt_object)
    timestamp = int(dt_object.timestamp() * rate)
    return timestamp


def get_random_headers():
    """
    返回一个包含随机 User-Agent 的请求头
    """
    ua = UserAgent()
    headers = {
        "User-Agent": ua.random,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    return headers


def initialize_ccxt_exchange(info: BasicInfo, rateLimit=False):
    retval: ccxt.Exchange = None
    if info.type == "spot":
        retval = getattr(ccxt, info.exchange)()
    elif info.type == "futures":
        retval = getattr(ccxt, info.exchange)({"options": {"defaultType": "future"}})
    else:
        raise ValueError(f"不支持的类型: {type}")
    retval.enableRateLimit = rateLimit
    return retval


def safe_load(value, default=None):
    return value if value else default
