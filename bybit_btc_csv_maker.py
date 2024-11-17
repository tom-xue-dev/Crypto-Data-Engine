import os
import random
from time import sleep, time
from enum import Enum

import pandas as pd
import requests

import utils

# 配置
FOLDER_NAME = "bybit_btc"
INTERVAL_TIME = {
    "1s": 1000,          # 1秒 = 1000毫秒
    "1m": 30000000,        
    "2m": 120000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1H": 3600000,
    "2H": 7200000,
    "4H": 14400000,
    "6H": 21600000,
    "8H": 28800000,
    "12H": 43200000,
    "1D": 86400000,
    "3D": 259200000,
    "1W": 604800000,
    "1M": 2592000000
}
SAVE_TIME = 100  # 每100次数据采集保存1次

COLUMNS = [
    'time',  # K线开盘时间
    'open',  # 开盘价
    'high',  # 最高价
    'low',   # 最低价
    'close', # 收盘价
    'volume' # 成交量
]

CURRENT_TIME = int(time() * 1000)

# 模式枚举
class Mode(Enum):
    CREATE = 0
    UPDATE = 1
    CONTINUE = 2

# 随机User-Agent列表
USER_AGENTS = [
    # 1-10: Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.126 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.5563.64 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.74 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.5304.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.103 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.102 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36",

    # 11-20: Firefox on Windows and macOS
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:99.0) Gecko/20100101 Firefox/99.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6; rv:97.0) Gecko/20100101 Firefox/97.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6; rv:96.0) Gecko/20100101 Firefox/96.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6; rv:93.0) Gecko/20100101 Firefox/93.0",

    # 21-30: Safari on macOS and iOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 13_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 12_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1 Mobile/15E148 Safari/604.1",

    # 31-40: Edge on Windows and macOS
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.126 Safari/537.36 Edg/113.0.1774.50",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36 Edg/112.0.1722.34",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.5563.64 Safari/537.36 Edg/111.0.1661.44",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36 Edg/110.0.1587.57",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.74 Safari/537.36 Edg/109.0.1518.61",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.124 Safari/537.36 Edg/108.0.1462.54",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.5304.107 Safari/537.36 Edg/107.0.1418.52",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.103 Safari/537.36 Edg/106.0.1370.34",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.102 Safari/537.36 Edg/105.0.1343.53",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36 Edg/104.0.1293.70",

    # 41-50: Opera, Brave, and Mobile Browsers
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.126 Safari/537.36 OPR/99.0.4664.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36 OPR/98.0.4756.82",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.5563.64 Safari/537.36 OPR/97.0.4686.71",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36 OPR/96.0.4686.63",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.74 Safari/537.36 OPR/95.0.4635.54",
    "Mozilla/5.0 (Linux; Android 11; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36 Brave/112.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.7 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.5 Mobile/15E148 Safari/604.1",
]


def _get_klines_data(to_time, interval, symbol="BTCUSDT", limit=500, retries=3, backoff=5):
    """
    发送GET请求并获取Bybit的K线数据
    :param to_time: 当前的to时间戳（毫秒）
    :param interval: 时间间隔，例如 '15m'
    :param symbol: 交易对，例如 'BTCUSDT'
    :param limit: 每次请求返回的数据量，最大500
    :param retries: 重试次数
    :param backoff: 重试间隔时间（秒）
    :return: 解析后的JSON数据
    """
    url = f"https://api2.bybit.com/spot/api/quote/v2/klines?symbol={symbol}&interval={interval}&limit={limit}&to={to_time}"
    for attempt in range(retries):
        try:
            headers = _get_random_headers()
            response = requests.get(url, headers=headers, timeout=15)  # 增加超时时间
            if response.status_code == 200:
                data = response.json()
                if data.get("ret_code") == 0:
                    return data.get("result", [])
                else:
                    print(f"API返回错误代码: {data.get('ret_code')}, 消息: {data.get('ret_msg')}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
        if attempt < retries - 1:
            print(f"等待 {backoff} 秒后重试...")
            sleep(backoff)
    print("多次请求失败，放弃此次请求。")
    return []

def _get_random_headers():
    """
    返回一个包含随机User-Agent的请求头
    """
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    return headers


def _save_new_data(new_data, file_name):
    """
    将新数据追加到CSV文件末尾
    :param new_data: 数据列表
    :param file_name: 输出CSV文件名
    """
    if not new_data:
        print("没有获取到任何数据。")
        return
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    try:
        # 将新数据转换为 DataFrame，并映射字段
        df_new = pd.DataFrame([
            {
                'time': item['t'],
                'open': item['o'],
                'high': item['h'],
                'low': item['l'],
                'close': item['c'],
                'volume': item['v']
            }
            for item in new_data
        ], columns=COLUMNS)
        
        # 检查是否所有必需的列都存在
        if not all(col in df_new.columns for col in COLUMNS):
            print("新数据缺少必要的列，跳过保存。")
            return
        
        # 数据清洗：转换数值列为浮点数，处理异常值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
        
        # 删除包含NaN值的行
        initial_length = len(df_new)
        df_new.dropna(subset=['time', 'open', 'high', 'low', 'close', 'volume'], inplace=True)
        cleaned_length = len(df_new)
        if cleaned_length < initial_length:
            print(f"清洗数据：删除了 {initial_length - cleaned_length} 行包含NaN值的数据。")
        
        # 将时间戳转换为系统时间
        df_new['time'] = df_new['time'].apply(lambda x: utils.timestamp_to_datetime(x))
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
        else:
            df = pd.DataFrame()
        updated_df = pd.concat([df, df_new], ignore_index=True)
        # 去重并排序
        updated_df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        updated_df.sort_values(by='time', ascending=True, inplace=True)
        # 保存更新后的数据回 CSV 文件
        updated_df.to_csv(file_name, index=False)
        print(f"数据已追加到 {file_name}")
    except PermissionError as pe:
        print(f"权限错误: 无法写入文件 {file_name}. 错误详情: {pe}")
    except Exception as e:
        print(f"保存数据到文件 {file_name} 时发生错误: {e}")

def _get_start_timestamp(filename, mode):
    """
    获取开始时间戳
    :param filename: CSV文件名
    :param mode: 模式，可选CREATE，CONTINUE，UPDATE
    :return: 开始时间戳
    """
    if mode == Mode.CREATE:
        return CURRENT_TIME
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if not df.empty:
                if mode == Mode.CONTINUE:
                    return utils.datetime_to_timestamp(df.iloc[0]['time'])  # 返回最后一条数据的时间戳
                if mode == Mode.UPDATE:
                    return utils.datetime_to_timestamp(df.iloc[-1]['time'])
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误: {e}")
    return None  # 如果文件不存在或为空，返回None

def make_csv(interval, mode=Mode.CREATE):
    file_name = f"{FOLDER_NAME}/{interval}.csv"
    if mode == Mode.CREATE and os.path.exists(file_name):
        print(f"{file_name}已存在，请使用CONTINUE或UPDATE模式")
        return
    new_data = []
    save_time = 0
    start_timestamp = _get_start_timestamp(file_name, mode)
    if start_timestamp is None:
        print(f"未找到文件{file_name}, 请先使用 CREATE 模式创建CSV")
        return
    end_time = start_timestamp
    # 根据模式判断从哪里开始请求数据
    if mode == Mode.CREATE:
        print(f"模式: CREATE - 从当前时间: {utils.timestamp_to_datetime(end_time)} 递减 {INTERVAL_TIME[interval]} 后请求")
    elif mode == Mode.CONTINUE and start_timestamp:
        print(f"模式: CONTINUE - 找到文件 {file_name}，从最后一个数据的时间戳: {utils.timestamp_to_datetime(start_timestamp)} 递减 {INTERVAL_TIME[interval]} 后请求")
    elif mode == Mode.UPDATE and start_timestamp:
        print(f"模式: UPDATE - 找到文件 {file_name}，从最新一个数据的时间戳: {utils.timestamp_to_datetime(start_timestamp)} 递增 {INTERVAL_TIME[interval]} 后请求")
    
    try:
        while True:
            if mode == Mode.UPDATE and end_time - INTERVAL_TIME[interval] > CURRENT_TIME:
                if new_data:
                    _save_new_data(new_data, file_name)
                print("数据已最新")
                break
            print(f"获取数据，to: {utils.timestamp_to_datetime(end_time)}")
            data = _get_klines_data(end_time, interval)
            if not data:
                if new_data:
                    _save_new_data(new_data, file_name)
                print("没有更多数据，停止请求。")
                break
            new_data.extend(data)
            # 更新end_time为上次获取数据的最后一个时间戳
            last_kline = data[-1]
            end_time = last_kline['t'] - 1
            if mode in [Mode.CREATE, Mode.CONTINUE]:
                end_time -= INTERVAL_TIME[interval]
            elif mode == Mode.UPDATE:
                end_time += INTERVAL_TIME[interval]
            # 可选：避免请求过于频繁，给服务器一些时间处理
            sleep(1)
            save_time += 1
            if save_time == SAVE_TIME:
                # 保存所有获取的数据到CSV文件
                _save_new_data(new_data, file_name)
                new_data = []
                save_time = 0
    except KeyboardInterrupt:
        print("\n手动停止数据采集。")
        if new_data:
            _save_new_data(new_data, file_name)
        else:
            print("没有获取到任何新数据。")


make_csv("1m", Mode.CONTINUE)