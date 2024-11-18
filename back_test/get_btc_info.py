import os
from datetime import datetime

import pandas as pd
from pathlib import Path


def get_target_file_path(level, exchange_name):
    """
    get the target btc file which stores the specific level
    param:
    level(str): represent the specific level that we want,e.g.15m
    exchange_name:represent the name of the crypto exchange institution,e.g.binance
    return:
    (str)target file os path if file exists,None otherwise.
    """
    path = ("1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h",
            "6h", "8h", "12h", "1d", "3d", "1mon")
    if level not in path:
        return None
    current_path = Path.cwd()
    target_path = current_path.parent / exchange_name / level
    if target_path.exists():
        return target_path
    else:
        return None


def get_btc_data(start_date, end_date, timeframe, exchange_name):
    """
    从编号文件中获取指定时间范围内的 BTC 数据。
    :param start_date: 起始日期 (datetime)
    :param end_date: 结束日期 (datetime)
    :param timeframe: 时间周期，例如 '15m'
    :param exchange_name: 交易所名称，例如 'binance'
    :return: 包含指定时间范围内的数据的 Pandas DataFrame
    """

    directory = get_target_file_path(timeframe, exchange_name)
    if directory is None:
        raise ValueError("can't find target directory")
    all_files = [file for file in os.listdir(directory) if file.startswith("part_") and file.endswith(".csv")]
    if not all_files:
        raise ValueError("no valid files")

    filtered_data = []
    for file in all_files:
        file_path = os.path.join(directory, file)

        # 读取文件内容
        df = pd.read_csv(file_path)
        # 确保日期列存在并解析为 datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])  # 转换为 datetime 对象
            # 筛选符合时间范围的数据
            df_filtered = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            if not df_filtered.empty:
                filtered_data.append(df_filtered)

    # 如果没有数据，返回空 DataFrame
    if not filtered_data:
        return pd.DataFrame()

    # 合并所有筛选后的数据
    result = pd.concat(filtered_data, ignore_index=True)
    return result



