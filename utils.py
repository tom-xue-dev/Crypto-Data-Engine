from datetime import datetime

import pytz


def timestamp_to_datetime(timestamp):
    """
    将时间戳（毫秒）转换为系统时间
    :param timestamp: 时间戳（毫秒）
    :return: 转换后的系统时间字符串
    """
    # 将毫秒级时间戳转换为秒级
    timestamp = timestamp / 1000
    # 使用datetime模块转换
    dt_object = datetime.utcfromtimestamp(timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M:%S')  # 格式化为“年-月-日 时:分:秒”

def datetime_to_timestamp(datetime_str):
    """
    将格式化后的datetime字符串转换为时间戳（毫秒）
    :param datetime_str: 格式化的datetime字符串，例如 '2024-11-15 14:30:00'
    :return: 对应的时间戳（毫秒）
    """
    # 定义输入的时间格式
    datetime_format = '%Y-%m-%d %H:%M:%S'
    # 将字符串转换为datetime对象（假设是UTC时间）
    utc_tz = pytz.timezone('UTC')
    dt_object = datetime.strptime(datetime_str, datetime_format)
    # 将datetime对象转换为UTC时区
    dt_object = utc_tz.localize(dt_object)
    # 获取时间戳（秒），然后乘以1000转换为毫秒
    timestamp = int(dt_object.timestamp() * 1000)
    return timestamp