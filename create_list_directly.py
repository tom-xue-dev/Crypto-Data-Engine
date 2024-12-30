import os
import pandas as pd
import pickle
from collections import defaultdict


def generate_nested_list_with_concat(start_time, end_time, asset_venue_list, folder_name):
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
    print(f"Processing folder: {folder_name}")
    print(f"Asset_venue list: {asset_venue_list}")
    print(f"Start time: {start_time}, End time: {end_time}")

    all_data = []

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
                print(f"Skipping unrelated file: {filename}")
                continue

            print(f"Processing file: {filename} (Asset_Venue: {asset_venue}, Time Frame: {time_frame})")
            file_path = os.path.join(folder_name, filename)

            # 加载文件数据
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                print(f"Loaded {len(data)} records from {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue

            # 转换为 DataFrame
            raw_data = []
            for record in data:
                time_val, values = record[0], record[1]
                if len(values) == 5:  # 确保格式正确
                    open_, high, low, close, asset_name = values

                    # 按时间范围过滤
                    if start_time <= time_val <= end_time:
                        raw_data.append([time_val, open_, high, low, close, asset_name])

            if raw_data:
                df = pd.DataFrame(raw_data, columns=["time", "open", "high", "low", "close", "asset_venue"])
                all_data.append(df)
                print(f"DataFrame from {filename} with shape: {df.shape} added to all_data.")

    # 合并所有数据
    if not all_data:
        print("No valid data found in the specified folder, assets, or time range.")
        return []

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")

    # 按时间戳分组并生成嵌套列表
    result = []
    for timestamp in combined_df["time"].unique():
        timestamp_data = combined_df[combined_df["time"] == timestamp]
        result.append([timestamp, timestamp_data[["asset_venue", "open", "high", "low", "close"]]])
        print(f"Added data for timestamp {timestamp} with shape: {timestamp_data.shape}")

    print(f"Processing completed, total timestamps: {len(result)}")
    return result


start_time = '2017-01-01'
end_time = '2024-12-31'
asset_list = ['1INCH-USDT_spot', '1INCH-USDT_future']  # 需要包含的资产
folder_name = "nested_pickle/15min"      # 存储 pkl 文件的文件夹

# 调用函数
nested_list = generate_nested_list_with_concat(start_time, end_time, asset_list, folder_name)

# 输出结果
for item in nested_list:
    timestamp, asset_df = item
    print(f"时间戳: {timestamp}")
    print(asset_df)