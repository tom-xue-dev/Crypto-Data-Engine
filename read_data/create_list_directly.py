import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd


def nested_list_to_list_of_dfs(nested_list):
    dfs = []
    for lis in nested_list:
        for row in lis:
            # row[0] -> 时间字符串
            # row[1] -> [open, high, low, close, asset]
            time_val = row[0]
            open_, high_, low_, close_, asset_ = row[1]

            # 创建一个只含 1 行的 DataFrame
            df = pd.DataFrame({
                'time': [time_val],
                'open': [open_],
                'high': [high_],
                'low': [low_],
                'close': [close_],
                'asset': [asset_]
            })

        # 将该 DataFrame 加入列表
            dfs.append(df)
    print(dfs[0:5])
    return dfs



def find_time_index(base_time: datetime, target_time: datetime, time_interval_minutes: timedelta):
    """
    查找目标时间在固定时间间隔序列中的下标。

    参数:
    - base_time_str: 基准时间
    - target_time_str: 目标时间
    - time_interval_minutes: 时间间隔 (int)，以分钟为单位

    返回:
    - 下标 (int)，如果目标时间合法
    - None，如果目标时间不合法
    """
    try:
        # 转换字符串为 datetime 对象

        # 固定时间间隔
        time_interval = time_interval_minutes
        # 检查目标时间是否早于基准时间
        if target_time < base_time:
            return None  # 目标时间早于基准时间，不合法
        # 计算时间差
        time_difference = target_time - base_time

        # 判断时间差是否是时间间隔的整数倍
        if time_difference % time_interval == timedelta(0):
            index = time_difference // time_interval
            return index
        else:
            return None  # 时间不对齐，不合法
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_data_lists(asset_venue_list, folder_name):
    """
    使用 pd.concat 将所有文件数据合并，并按时间戳生成嵌套列表。
    文件命名格式为 `asset_venue_time.pkl`，例如 `1INCH-USDT_spot_15min.pkl`。

    参数:
    - start_time: 起始时间戳 (str)，格式如 '2024-01-01'
    - end_time: 结束时间戳 (str)，格式如 '2024-12-31'
    - asset_venue_list: 资产与场地组合列表 (list)，如 ['1INCH-USDT_spot', '1INCH-USDT_future']
    - folder_name: 文件夹名称 (str)，存储多个资产的 pkl 文件

    返回:
    - 嵌套列表，外层为时间戳，内层为多个资产的合并 DataFrame
    """

    data = [[]]
    i = 0
    # 遍历匹配的文件
    for filename in os.listdir(folder_name):
        if filename.endswith(".pkl"):
            parts = filename.replace(".pkl", "").split("_")
            if len(parts) < 3:
                print(f"Skipping invalid file: {filename}")
                continue

            asset_venue = "_".join(parts[:2])  # 提取 asset_venue
            time_frame = parts[2]  # 提取时间分辨率，如 15min、1h 等

            # 检查是否为目标资产与场地
            if asset_venue not in asset_venue_list:
                # print(f"Skipping unrelated file: {filename}")
                continue

            print(f"Processing file: {filename} (Asset_Venue: {asset_venue}, Time Frame: {time_frame})")
            file_path = os.path.join(folder_name, filename)

            # 加载文件数据

            try:
                with open(file_path, "rb") as f:
                    data[i] = pickle.load(f)
                    i += 1
                    data.append([])
                print(f"Loaded {len(data)} records from {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue
    del data[i]
    return data


def filter_crypt_data(
    start_time: datetime,
    end_time: datetime,
    interval: int,
    data_frames: list
) -> list:
    """
    返回: final_lists
       final_lists[i] -> DataFrame，含该时间戳下所有资产的多行记录
    """
    interval_td = timedelta(minutes=interval)
    list_len = (end_time - start_time) // interval_td + 1

    final_lists = [pd.DataFrame() for _ in range(list_len)]  # 先占位

    for df in data_frames:
        print("123")
        # 确保 time 是 datetime
        if df["time"].dtype == object:
            df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        df = df.sort_values("time").reset_index(drop=True)
        df_start_time = df.loc[0, "time"]
        if pd.isna(df_start_time):
            continue

        # same logic
        if df_start_time <= start_time:
            start_idx = find_time_index(df_start_time, start_time, interval_td)
            if start_idx is None:
                continue
            for i in range(list_len):
                row_idx = start_idx + i
                if row_idx < df.shape[0]:
                    # 取该行
                    row_data = df.iloc[[row_idx]]  # 用[[row_idx]]取出DataFrame格式
                    # 拼接到 final_lists[i]
                    final_lists[i] = pd.concat([final_lists[i], row_data], ignore_index=True)
        else:
            start_idx = find_time_index(start_time, df_start_time, interval_td)
            if start_idx is None:
                continue
            for i in range(start_idx, list_len):
                row_idx = i - start_idx
                if row_idx < df.shape[0]:
                    row_data = df.iloc[[row_idx]]
                    final_lists[i] = pd.concat([final_lists[i], row_data], ignore_index=True)

    return final_lists

if __name__ == "__main__":
    start_time = datetime.strptime('2020-12-26 05:00:00', "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime('2023-12-25 05:15:01', "%Y-%m-%d %H:%M:%S")
    asset_list = ['ADA-USDT_spot']  # 需要包含的资产
    script_path = Path(__file__).resolve()
    base_path = script_path.parents[1] / "data"

    folder_name = base_path / "nested_pickle/1min"  # 存储 pkl 文件的文件夹

    # 调用函数
    nested_list = get_data_lists(asset_list, folder_name)
    print("start")
    nested_list= nested_list_to_list_of_dfs(nested_list)
    print("over")
    for i in range(5):
        print(nested_list[0][i])
    start_t = time.time()
    result_data = filter_crypt_data(start_time, end_time, 1, nested_list)
    end_t = time.time()
    print(end_t-start_t)

    # print(f"Execution time: {end - start:.6f} seconds")
