import datetime
import mmap
import os
import pickle
import random
import time

import pandas as pd
from pathlib import Path
from datetime import datetime


def load_filtered_data_as_list(start_time, end_time, asset_list, level: str):
    """
    按指定的时间范围和资产列表从月度 Pickle 文件中读取并过滤数据，返回列表。

    参数
    ----------
    start_time : str 或 pandas.Timestamp 或 datetime.datetime
        数据读取的起始时间。
    end_time : str 或 pandas.Timestamp 或 datetime.datetime
        数据读取的结束时间。
    asset_list : list
        需要读取的资产列表。例如：['BTC-USDT_spot', 'ETH-USDT_future']
    level : str
        资产的时间级别 可以选择"1d,1min,15min"

    返回
    ----------
    filtered_list : list
        包含多个过滤后的 DataFrame 的列表。
    """
    # 1. 转换 start_time 和 end_time 为 pandas.Timestamp
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # 2. 生成需要读取的月份列表
    # 使用 PeriodRange 生成所有覆盖的月份
    all_periods = pd.period_range(start=start_time, end=end_time, freq='M')

    filtered_list = []  # 用于存储所有符合条件的 DataFrame

    for period in all_periods:
        # 构造文件名，例如 "2023-01.pkl"
        filename = f"{period.year}-{period.month:02d}.pkl"
        script_path = Path(__file__).resolve()
        base_path = script_path.parents[1] / "data" / "data_divided_by_month" / level
        print(base_path)
        pickle_path = os.path.join(base_path, filename)

        if not os.path.exists(pickle_path):
            print(f"警告：文件不存在 {pickle_path}，跳过。")
            continue  # 跳过不存在的文件

        # 3. 加载 Pickle 文件
        with open(pickle_path, 'rb') as f:
            month_list = pickle.load(f)  # 这是一个包含多个 DataFrame 的列表

        print(f"已加载文件 {filename}，包含 {len(month_list)} 个时间戳的数据。")

        # 4. 遍历每个时间戳的 DataFrame 进行过滤
        for df in month_list:
            # 假设每个 DataFrame 代表一个时间戳的数据，且 'time' 列为时间戳
            if df.empty:
                continue  # 跳过空的 DataFrame

            # 获取当前 DataFrame 的时间戳（假设所有行的时间戳相同）
            current_time = df['time'].iloc[0]

            # 过滤时间范围
            if not (start_time <= current_time <= end_time):
                continue  # 跳过不在时间范围内的数据

            # 过滤资产列表
            if asset_list:
                filtered_df = df[df['asset'].isin(asset_list)]
            else:
                filtered_df = df  # 如果 asset_list 为空，则不过滤资产

            if not filtered_df.empty:
                filtered_list.append(filtered_df)

    if not filtered_list:
        print("未找到符合条件的数据。")
        return []  # 返回空列表

    print(f"返回的列表包含 {len(filtered_list)} 个过滤后的 DataFrame。")
    return filtered_list


def map_and_load_pkl_files(level: str, asset_list=None, start_time=None, end_time=None):
    # 获取文件列表，按名称排序（假设文件名格式为 YYYY-MM.pkl）
    script_path = Path(__file__).resolve()
    folder_path = script_path.parents[1] / "data" / "big_files" / level

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pkl')])

    data = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "rb") as f:
            # 映射文件到内存
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # 使用 pickle 加载数据
            loaded_data = pickle.loads(mm[:])
            mm.close()

            # 现在 loaded_data 是一个 MultiIndex DataFrame
            if isinstance(loaded_data, pd.DataFrame):  # 确保它是 DataFrame
                # 检查 MultiIndex 是否存在
                if isinstance(loaded_data.index, pd.MultiIndex):
                    df = loaded_data
                    # 过滤时间范围
                    if start_time and end_time:
                        # 假设 time 列是索引的第一层
                        df = df[(df.index.get_level_values('time') >= start_time) &
                                (df.index.get_level_values('time') <= end_time)]

                    # 过滤资产列表
                    if asset_list:
                        df = df[df.index.get_level_values('asset').isin(asset_list)]

                    # 如果 DataFrame 仍然有数据，则加入结果列表
                    if not df.empty:
                        data.append((file_name, df))
                    else:
                        pass
                else:
                    print(f"警告：{file_name} 的索引不是 MultiIndex 类型。")
            else:
                print(f"警告：{file_name} 的数据不是 DataFrame 类型。")

    # 拼接所有符合条件的 DataFrame（保持 MultiIndex）
    if data:
        combined_df = pd.concat([df for _, df in data], axis=0)
        return combined_df
    else:
        print("没有符合条件的数据。")
        return pd.DataFrame()  # 返回一个空的 DataFrame


def select_assets(future=False, spot=False, n=5, m=None, start_time=None):
    """
    从资产表中根据参数随机选择资产。

    参数:
        future (bool): 是否从 future 列中选择
        spot (bool): 是否从 spot 列中选择
        n (int): 需要选择的资产数量

    返回:
        list: 随机选择的资产列表
    """
    script_path = Path(__file__).resolve()
    file = script_path.parents[1] / "data" / "big_files" / 'market_cap.pkl'
    with open(file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        loaded_data = pickle.loads(mm[:])
        mm.close()

    if not m is None:
        return loaded_data.loc[start_time].nlargest(m, "market_cap").index.values
    else:
        return loaded_data.loc[start_time].sample(n=min(n, len(loaded_data.loc[start_time]))).index.values


if __name__ == "__main__":
    # 定义读取参数
    start = "2022-12-01"
    end = "2024-12-31"
    asset = select_assets(spot=True, start_time=start,m=5)
    print(asset)
