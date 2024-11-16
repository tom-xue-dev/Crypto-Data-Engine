import os
from time import sleep, time
from enum import Enum

import pandas as pd
import requests

import utils


FOLDER_NAME = "binance_btc"  
INTERVAL_TIME = {
    "1s": 1000000,
    "1m": 60000000,
    "2m": 120000000,
    "3m": 180000000,
    "5m": 300000000,
    "15m": 900000000,
    "30m": 1800000000,
    "1H": 3600000000,
    "2H": 7200000000,
    "4H": 14400000000,
    "6H": 21600000000,
    "8H": 28800000000,
    "12H": 43200000000,
    "1D": 86400000000,
    "3D": 259200000000,
    "1W": 604800000000,
    "1M": 2592000000000
}
CURRENT_TIME = int(time() * 1000)
COLUMNS = [
    'time', # k线开盘时间
    'open', # 开盘价
    'high', # 最高价
    'low', # 最低价
    'close', # 收盘价
    'volume', # 成交量
    'close_time', # k线收盘时间
    'transaction_volume', # 成交额 
    'transaction_number', # 成交笔数
    'active_buying_volume', # 主动买入成交量
    'active_buying_transaction_volume', # 主动买入成交额
    'ignore' # 忽略
]
SAVE_TIME = 5 # 每5次数据采集保存1次
# 模式枚举
class Mode(Enum):
    CREATE = 0
    UPDATE = 1
    CONTINUE = 2


def _get_klines_data(end_time, interval, symbol="BTCUSDT", limit=1000):
    """
    发送GET请求并获取K线数据
    :param end_time: 当前的endTime
    :param interval: 时间间隔，例如 '15m'
    :param symbol: 交易对，例如 'BTCUSDT'
    :param limit: 每次请求返回的数据量，最大1000
    :return: 解析后的JSON数据
    """
    url = f"https://api.binance.com/api/v3/uiKlines?endTime={end_time}&limit={limit}&symbol={symbol}&interval={interval}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return []

def _save_new_data(new_data, file_name):
    """
    将数据追加到CSV文件头部
    :param new_data: 数据列表
    :param file_name: 输出CSV文件名
    """
    if not new_data:
        print("没有获取到任何数据。")
        return
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    # 读取现有 CSV 文件
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame()  # 如果文件不存在，则初始化为空的 DataFrame
    # 将新数据转换为 DataFrame
    new_df = pd.DataFrame(new_data, columns=COLUMNS)
    # 将时间戳转换为系统时间
    new_df['time'] = new_df['time'].apply(lambda x: utils.timestamp_to_datetime(x))
    # 将新数据插入到现有数据的头部
    updated_df = pd.concat([new_df, df], ignore_index=True)
    # 去重并排序
    updated_df.drop_duplicates(subset=['time'], keep='first', inplace=True)
    updated_df.sort_values(by='time', ascending=False, inplace=True)
    # 保存更新后的数据回 CSV 文件
    updated_df.to_csv(file_name, index=False)
    print(f"数据已保存到 {file_name}")

def _get_start_timestamp(filename, mode):
    """
    获取开始时间戳
    :param filename：CSV文件名
    :param mode：模式，可选CREATE，CONTINUE，UPDATE
    :return: 开始时间戳
    """
    if mode == Mode.CREATE:
        return CURRENT_TIME
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if not df.empty:
            if mode == Mode.CONTINUE:
                return utils.datetime_to_timestamp(df.iloc[-1]['time'])  # 返回最后一条数据的时间戳
            if mode == Mode.UPDATE:
                return utils.datetime_to_timestamp(df.iloc[0]['time'])
    return None  # 如果文件不存在或为空，返回None

def make_csv(interval, mode=Mode.CREATE):
    file_name = f"{FOLDER_NAME}/{interval}.csv"
    if mode == Mode.CREATE and os.path.exists(file_name):
        print(f"{file_name}已存在，请使用CONTINUE或UPDATE模式")
        return
    new_data = []
    save_time = 0
    start_timestamp = _get_start_timestamp(file_name,mode)
    end_time = start_timestamp
    # 根据模式判断从哪里开始请求数据
    if mode == Mode.CREATE:
        print(f"模式: CREATE - 从当前时间: {start_timestamp} 递减 {INTERVAL_TIME[interval]} 后请求")
    elif mode == Mode.CONTINUE and start_timestamp:
        print(f"模式: CONTINUE - 找到文件 {file_name}，从最后一个数据的时间戳: {start_timestamp} 递减 {INTERVAL_TIME[interval]} 后请求")
    elif mode == Mode.UPDATE and start_timestamp:
        print(f"模式: UPDATE - 找到文件 {file_name}，从最新一个数据的时间戳: {start_timestamp} 递增 {INTERVAL_TIME[interval]} 后请求")
    try:
        while True:
            if end_time - INTERVAL_TIME[interval] > CURRENT_TIME:
                if new_data != []:
                    _save_new_data(new_data, file_name)
                print(f"数据已最新")
                break
            print(f"获取数据，endTime: {utils.timestamp_to_datetime(end_time)}")
            data = _get_klines_data(end_time, interval)
            if not data:
                print("没有更多数据，停止请求。")
                break
            new_data.extend(data)
            # 更新endTime为上次获取数据的第一个时间戳
            end_time = data[-1][0] - 1
            if mode in [Mode.CREATE, Mode.CONTINUE]:
                end_time -= INTERVAL_TIME[interval]
            elif mode == Mode.UPDATE:
                end_time += INTERVAL_TIME[interval]
            # 可选：避免请求过于频繁，给服务器一些时间处理
            sleep(1)
            save_time = save_time + 1
            if save_time == SAVE_TIME:
                # 保存所有获取的数据到CSV文件
                _save_new_data(new_data, file_name)
                new_data = []
                save_time = 0
    except KeyboardInterrupt:
        print("\n手动停止数据采集。")
        _save_new_data(new_data, file_name)

if __name__ == "__main__":
    make_csv("1s")
