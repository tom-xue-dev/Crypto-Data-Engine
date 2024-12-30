import os
import pickle
import pandas as pd
import gc

def build_time_grouped_list_no_temp_minimal(folder_path, assets, start_time, end_time):
    """
    优化思路示例：
      1) 使用 list 收集各个文件的 DF，最后一次性 concat。
      2) 基于文件名快速过滤。
      3) 删除不需要的列（如果有多余列）。
    """
    # 收集所有文件处理后的 DF
    list_of_dfs = []

    for file_name in os.listdir(folder_path):
        # 1) 只读取 pkl
        if not file_name.endswith('.parquet'):
            continue

        # 2) 如果提供了 assets，则基于文件名做快速过滤
        if assets and not any(asset in file_name for asset in assets):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"正在读取文件: {file_name}")
        try:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
        except Exception as e:
            print(f"无法读取文件 {file_name}，错误: {e}")
            continue

        # 3) 必要列检查

        # 4) time 转为 datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)

        # 5) 时间过滤
        df = df[df['time'].between(start_time, end_time)]

        # 6) 行级资产过滤
        if assets:
            df = df[df['asset'].isin(assets)]

        # ==== 如果这里还有多余的列，可以顺手干掉，如 ====
        # keep_cols = ['time', 'open', 'high', 'low', 'close', 'asset']
        # df = df[keep_cols]

        # 收集到列表
        list_of_dfs.append(df)

        del df
        gc.collect()

    # 一次性拼接
    if len(list_of_dfs) == 0:
        # 无数据时直接返回空列表
        return []

    df_accumulated = pd.concat(list_of_dfs, ignore_index=True)
    del list_of_dfs
    gc.collect()

    # 按 time + asset 排序（如果必须）
    # df_accumulated.sort_values(["time", "asset"], inplace=True, ignore_index=True)

    # 最后 groupby
    time_grouped_list = []
    for time_value, group_df in df_accumulated.groupby("time"):
        time_grouped_list.append((time_value, group_df))

    return time_grouped_list


def build_time_grouped_list_no_temp_minimal_parquet(folder_path, assets, start_time, end_time):
    """
    读取指定文件夹下的所有 Parquet 文件：
      1) 使用 list 收集各个文件的 DataFrame，最后一次性 concat。
      2) 基于文件名快速过滤（基于资产）。
      3) 确保必要的列存在。
      4) 过滤时间范围和资产。
      5) 按 time 分组，返回列表。

    参数:
        folder_path (str): Parquet 文件所在文件夹路径。
        assets (list[str]): 需要的资产列表。若为空列表则表示不基于文件名的资产过滤。
        start_time (str): 时间段起始字符串，例如 '2017-01-01 00:00:00'。
        end_time (str):   时间段结束字符串，例如 '2024-12-13 00:00:00'。

    返回:
        list[tuple[pd.Timestamp, pd.DataFrame]]:
            [(time1, df_group1), (time2, df_group2), ...]
    """
    # 将字符串的起止时间转换为 Timestamp，以便比较
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    # 用于存放所有拼接后的数据
    list_of_dfs = []

    # 定义必要的列
    required_cols = {'time', 'open', 'high', 'low', 'close', 'asset'}

    # 遍历文件夹下所有 Parquet 文件
    for file_name in os.listdir(folder_path):
        # 1) 只读取 .parquet 文件
        if not file_name.endswith('.parquet'):
            continue

        # 2) 如果提供了 assets，则基于文件名做快速过滤
        if assets and not any(asset in file_name for asset in assets):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"正在读取文件: {file_name}")

        # 3) 尝试读取当前文件
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"无法读取文件 {file_name}，错误: {e}")
            continue

        # 4) 检查必要列
        if not required_cols.issubset(df.columns):
            print(f"文件 {file_name} 中缺少必要列 {required_cols - set(df.columns)}，跳过。")
            continue

        # 5) 确保 time 列为 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            try:
                df['time'] = pd.to_datetime(df['time'])
            except Exception as e:
                print(f"无法将 'time' 列转换为 datetime 类型，文件 {file_name}，错误: {e}")
                continue

        # 6) 时间过滤
        df = df[df['time'].between(start_dt, end_dt)]

        # 7) 行级资产过滤
        if assets:
            df = df[df['asset'].isin(assets)]

        # 8) 如果需要，删除多余的列
        # keep_cols = ['time', 'open', 'high', 'low', 'close', 'asset']
        # df = df[keep_cols]

        # 9) 收集到列表
        list_of_dfs.append(df)

        # 释放内存
        del df
        gc.collect()

    # 一次性拼接
    if not list_of_dfs:
        # 无数据时直接返回空列表
        print("没有有效的数据被读取。")
        return []

    try:
        df_accumulated = pd.concat(list_of_dfs, ignore_index=True)
    except ValueError as e:
        print(f"数据合并失败，错误: {e}")
        return []

    # 释放列表内存
    del list_of_dfs
    gc.collect()


    # 按 time + asset 排序（如果必须）
    df_accumulated.sort_values(["time", "asset"], inplace=True, ignore_index=True)
    # 按 time 分组，生成 [(time_value, group_df), ...] 列表
    time_grouped_list = []
    for time_value, group_df in df_accumulated.groupby("time"):
        time_grouped_list.append((time_value, group_df))

    return time_grouped_list

# 使用示例
if __name__ == "__main__":
    folder_path = "data_parquet/1min"       # 假设文件夹内存放若干 15m pkl 文件
    # 如果 assets 为空 [], 表示无需基于文件名筛选任何资产，则读取所有文件
    assets = [
              "ADA-USDT_spot", "ADA-USDT_future",
              "HIVE-USDT_spot", "HIVE-USDT_future"
              ]     # 举例
    start_time = "2017-01-01 00:00:00"
    end_time = "2024-12-13 00:00:00"

    grouped_data_list = build_time_grouped_list_no_temp_minimal_parquet(
        folder_path=folder_path,
        assets=assets,
        start_time=start_time,
        end_time=end_time
    )

    print(f"总共得到 {len(grouped_data_list)} 个按时间分组的结果。")
    # 例如查看第一个分组
    print(grouped_data_list[0:3])
    print(grouped_data_list[-2])

