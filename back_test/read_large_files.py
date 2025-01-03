import os
import pickle
import pandas as pd
from pathlib import Path


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
        base_path = script_path.parents[1] / "data" / "data_divided_by_mouth"/level
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


if __name__ == "__main__":
    # 定义读取参数
    start_time = "2024-12-01"
    end_time = "2024-12-31"
    asset_list = ["1MBABYDOGE-USDT_spot", "AAVE-USDT_future"]  # 替换为您需要的资产


    # 调用函数读取并过滤数据
    filtered_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "15min")
