from pathlib import Path
import os
import pandas as pd
import glob


def merge_csv_to_pickle(csv_folder, pickle_file, asset_name):
    """
    将指定文件夹内的所有 CSV 文件合并为一个 Pickle 文件，
    并添加一列名为 'asset'，值为指定名称。

    参数:
        csv_folder (str): 包含 CSV 文件的文件夹路径。
        pickle_file (str): 输出的 Pickle 文件路径。
        asset_name (str): 插入 'asset' 列的内容。
    """

    # 检查文件夹是否存在
    if not Path(csv_folder).exists():
        print(f"文件夹不存在: {csv_folder}, 跳过！")
        return

    # 获取文件夹中所有 CSV 文件路径
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    if not csv_files:
        print(f"文件夹中没有 CSV 文件: {csv_folder}, 跳过！")
        return

    # 初始化一个空列表存储所有 DataFrame
    data_list = []

    # 遍历所有 CSV 文件并处理
    for file in csv_files:
        try:
            if os.path.getsize(file) > 0:  # 检查文件是否为空
                df = pd.read_csv(file)  # 读取 CSV 文件
                if not df.empty:  # 确保 DataFrame 非空
                    df['asset'] = asset_name  # 添加 'asset' 列
                    data_list.append(df)
                else:
                    print(f"警告: 文件 {file} 内容为空，已跳过！")
            else:
                print(f"警告: 文件 {file} 是空文件，已跳过！")
        except Exception as e:
            print(f"错误: 读取文件 {file} 时出错，错误信息: {e}")

    # 合并所有 DataFrame
    if data_list:
        merged_data = pd.concat(data_list, ignore_index=True)
        # 将合并后的数据保存为 Pickle 文件
        merged_data.to_pickle(pickle_file)
        print(f"数据已成功保存为 Pickle 文件: {pickle_file}")
    else:
        print("没有有效的 CSV 文件，未生成 Pickle 文件。")


import os
import glob
from pathlib import Path
import pandas as pd

def merge_csv_to_parquet(csv_folder, parquet_file, asset_name):
    """
    将指定文件夹内的所有 CSV 文件合并为一个 Parquet 文件，
    并添加一列名为 'asset'，值为指定名称。优化数据类型以减少内存占用。

    参数:
        csv_folder (str or Path): 包含 CSV 文件的文件夹路径。
        parquet_file (str or Path): 输出的 Parquet 文件路径。
        asset_name (str): 插入 'asset' 列的内容。
    """

    # 检查文件夹是否存在
    csv_folder = Path(csv_folder)
    if not csv_folder.exists() or not csv_folder.is_dir():
        print(f"文件夹不存在或不是目录: {csv_folder}, 跳过！")
        return

    # 获取文件夹中所有 CSV 文件路径
    csv_files = list(csv_folder.glob("*.csv"))
    if not csv_files:
        print(f"文件夹中没有 CSV 文件: {csv_folder}, 跳过！")
        return

    data_list = []

    # 遍历所有 CSV 文件并处理
    for file in csv_files:
        try:
            if file.stat().st_size > 0:  # 检查文件是否为空
                df = pd.read_csv(file)
                if not df.empty:
                    df['asset'] = asset_name  # 添加 'asset' 列

                    # 数据类型优化
                    # 1. 将 'asset' 列转换为分类类型
                    if 'asset' in df.columns:
                        df['asset'] = df['asset'].astype('category')

                    # 2. 优化数值类型
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    for col in numeric_cols:
                        if pd.api.types.is_integer_dtype(df[col]):
                            df[col] = pd.to_numeric(df[col], downcast='integer')  # 转换为 int32 或更小
                        elif pd.api.types.is_float_dtype(df[col]):
                            df[col] = pd.to_numeric(df[col], downcast='float')  # 转换为 float32

                    data_list.append(df)
                else:
                    print(f"警告: 文件 {file} 内容为空，已跳过！")
            else:
                print(f"警告: 文件 {file} 是空文件，已跳过！")
        except Exception as e:
            print(f"错误: 读取文件 {file} 时出错，错误信息: {e}")

    # 合并所有 DataFrame 并保存到 Parquet
    if data_list:
        merged_data = pd.concat(data_list, ignore_index=True)

        # 将合并后的数据保存为 Parquet 文件
        # 可以根据需要为 Parquet 指定压缩方式，比如 compression='snappy'
        try:
            merged_data.to_parquet(parquet_file, index=False, compression='snappy')
            print(f"数据已成功保存为 Parquet 文件: {parquet_file}")
        except Exception as e:
            print(f"错误: 保存 Parquet 文件 {parquet_file} 时出错，错误信息: {e}")
    else:
        print("没有有效的 CSV 文件，未生成 Parquet 文件。")



def list_subdirectories(folder_path):
    """
    列出指定路径下的所有一级文件夹名。
    参数:
        folder_path (str): 目标路径。
    返回:
        list: 包含所有一级文件夹名的列表。
    """
    # 使用 Path 对象进行过滤
    return [item.name for item in Path(folder_path).iterdir() if item.is_dir()]


# 示例调用
venue = "future"
level = '1d'
folder_path = f"binance/{venue}"
subdirs = list_subdirectories(folder_path)

# 示例调用
for asset in subdirs:
    # 根据 asset 构造路径
    csv_folder_path = f"binance/{venue}/{asset}/{level}"
    output_pickle_path = f"data_pickle/1d/{asset}_{venue}_{level}.parquet"
    asset_name = f"{asset}_{venue}"  # 指定 'asset' 列的值
    merge_csv_to_parquet(csv_folder_path, output_pickle_path, asset_name)
