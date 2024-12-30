import os
import pickle
import pandas as pd
import gc

def build_nested_list_from_parquet(folder_path,
                                   assets,
                                   start_time,
                                   end_time,
                                   temp_file="temp.pkl"):
    """
    根据文件夹名称自动推断 freq，并在 DataFrame 行级数据筛选时使用资产信息 (assets)。
    - assets = [] => 读取文件夹下所有 .parquet 文件
    - assets = 'spot' => 只读取文件名中包含 '_spot_' 的 .parquet 文件
    - assets = 'futures' => 只读取文件名中包含 '_futures_' 的 .parquet 文件
    - assets = [实际资产列表] => 读取文件后，再对 df['asset'] 做 isin(assets) 的行级过滤

    文件夹名示例：
        folder_path = "data_parquet/1d"  => freq = "1d"
        folder_path = "data_parquet/15m" => freq = "15min"
        folder_path = "data_parquet/1m"  => freq = "1min"
    """
    # 从文件夹名称自动推断 freq
    freq = os.path.basename(folder_path)

    # 1) 若存在断点文件则加载
    if os.path.exists(temp_file):
        print(f"检测到临时文件 {temp_file}，加载中...")
        with open(temp_file, 'rb') as f:
            result_dict = pickle.load(f)
        print(f"临时文件加载成功，将从中断位置继续。")
    else:
        # 初始化结果字典：按时间戳组织数据
        time_range = pd.date_range(start=start_time, end=end_time, freq=freq)
        time_range_str = time_range.strftime('%Y-%m-%d %H:%M:%S').tolist()
        result_dict = {time_str: {} for time_str in time_range_str}

    processed_files = set(result_dict.get("_processed_files", []))

    # 2) 遍历文件夹内的 .parquet 文件
    for file_name in os.listdir(folder_path):
        # 只处理 .parquet 文件，排除已处理过的
        if not file_name.endswith('.parquet'):
            continue
        if file_name in processed_files:
            continue

        # 根据 assets 决定是否处理该文件
        should_process_file = False

        if isinstance(assets, list) and len(assets) == 0:
            # 空列表 => 不限制
            should_process_file = True
        elif assets == "spot":
            if "_spot_" in file_name:
                should_process_file = True
        elif assets == "future":
            if "_future_" in file_name:
                should_process_file = True
        elif isinstance(assets, list) and len(assets) > 0:
            # 例：假设文件名形如 "BTC-USDT_15m.parquet", "ETH-USDT_1d.parquet" 等
            # 若文件名里没有任何目标资产，则直接跳过
            if any(asset in file_name for asset in assets):
                should_process_file = True
            else:
                # 如果这里直接 continue，就可以在读文件前跳过
                continue
        else:
            # 其他情况(比如 assets 不是预期的格式)，跳过
            continue

        if not should_process_file:
            continue

        # 3) 读取文件并处理数据
        file_path = os.path.join(folder_path, file_name)
        print(f"正在处理文件: {file_name}")

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"无法读取文件 {file_name}，错误: {e}")
            continue

        required_cols = {'time', 'open', 'high', 'low', 'close', 'asset'}
        if not required_cols.issubset(df.columns):
            print(f"文件 {file_name} 中缺少必要列，跳过。")
            continue

        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)

        # 时间段过滤
        df = df[df['time'].between(start_time, end_time)]

        # 如果 assets 是一个真实的资产列表，则再对 asset 列做过滤
        if isinstance(assets, list) and len(assets) > 0:
            df = df[df['asset'].isin(assets)]

        # 填充到 result_dict
        for _, row in df.iterrows():
            t_str = row['time'].strftime('%Y-%m-%d %H:%M:%S')
            if t_str in result_dict:
                result_dict[t_str][row['asset']] = [
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['asset'],
                ]

        # 标记已处理
        processed_files.add(file_name)
        result_dict["_processed_files"] = list(processed_files)

        # 保存中间结果到临时文件
        with open(temp_file, 'wb') as temp_f:
            pickle.dump(result_dict, temp_f)
        print(f"已保存中间结果到临时文件 {temp_file}")

        # 释放内存
        del df
        gc.collect()

    # 4) 去掉中间标记并构建嵌套列表
    if "_processed_files" in result_dict:
        del result_dict["_processed_files"]

    nested_list = []
    for time_str, asset_dict in result_dict.items():
        if asset_dict:  # 若该时间戳下有数据
            nested_list.append([time_str] + list(asset_dict.values()))

    return nested_list


def generate_nested_dataframes(folder_path,
                               assets,
                               start_time,
                               end_time,
                               temp_file="temp.pkl"):
    """
    从文件夹名称自动推断 freq，并生成每个时间戳对应的 DataFrame 列表。

    参数:
        folder_path (str):
          e.g. 'data_parquet/1d', 'data_parquet/15m', 'data_parquet/1m'
        assets (list/str):
          - 为空列表 [] => 读取所有 .parquet
          - 'spot' => 文件名需含 '_spot_'
          - 'futures' => 文件名需含 '_futures_'
          - [具体资产列表] => 读取后，对 df['asset'].isin(assets) 过滤
        start_time (str): 时间段起始，如 '2017-01-01 00:00:00'
        end_time (str):   时间段结束，如 '2024-12-13 00:00:00'
        temp_file (str):  临时文件，用于断点续作

    返回:
        list of pd.DataFrame: 每个 DataFrame 对应一个时间戳的数据
    """
    # 1) 构建嵌套列表
    nested_list = build_nested_list_from_parquet(
        folder_path=folder_path,
        assets=assets,
        start_time=start_time,
        end_time=end_time,
        temp_file=temp_file
    )

    # 2) 将每个外层元素转换为 DataFrame 并收集到列表中
    dataframes_list = []
    for row in nested_list:
        time_str = row[0]
        records = row[1:]  # 各资产的数据

        # 构建 DataFrame
        df = pd.DataFrame(records, columns=["open", "high", "low", "close", "asset"])
        df.insert(0, "time", pd.to_datetime(time_str))  # 添加时间列

        dataframes_list.append(df)

    return dataframes_list

# 示例用法
if __name__ == "__main__":
    folder = "data_parquet/1min"
    assets = ["ADA-USDT_future", "ADA-USDT_spot"]  # 或者使用 'spot', 'future', []
    start = "2017-01-01 00:00:00"
    end = "2024-12-29 23:59:59"

    dfs = generate_nested_dataframes(
        folder_path=folder,
        assets=assets,
        start_time=start,
        end_time=end,
        temp_file="temp.pkl"
    )

    # 示例：打印前5个 DataFrame
    for i, df in enumerate(dfs[:5]):
        print(f"DataFrame {i+1}:")
        print(df)




