import glob
import matplotlib.pyplot as plt
import os
import pandas as pd


def load_dataset_from_folder(folder_path):
    """
    从指定文件夹加载所有数据文件并合并为一个 DataFrame

    :param folder_path: str, 数据文件所在文件夹路径
    :return: pd.DataFrame, 合并后的数据集
    """
    # 获取文件夹中所有 CSV 文件的路径
    file_pattern = os.path.join(folder_path, "*.csv")  # 如果文件格式不是 CSV，请修改为对应格式
    file_list = glob.glob(file_pattern)

    if not file_list:
        raise FileNotFoundError(f"No files found in folder: {folder_path}")

    # 读取所有文件并合并
    dataframes = []
    for file in file_list:
        print(f"Loading file: {file}")
        df = pd.read_csv(file)  # 根据文件格式调整读取方法
        dataframes.append(df)

    # 合并所有 DataFrame
    combined_dataset = pd.concat(dataframes, ignore_index=True)
    return combined_dataset


def calculate_avg_change_next_n_periods(data, n, time_column='time', price_column='close'):
    """
    计算所有时间点后 n 单位时间内的平均涨跌幅，以及胜率。

    参数:
        data (pd.DataFrame): 包含时间序列和价格数据的 DataFrame。
        n (int): 后 n 单位时间的窗口大小。
        time_column (str): 时间列的名称，默认为 'Date'。
        price_column (str): 价格列的名称，默认为 'Price'。

    返回:
        pd.DataFrame: 包含每个时间点后 n 单位时间平均涨跌幅，以及胜率。
        float: 胜率（涨幅为正的比例）。
    """
    # 检查输入数据是否包含必要的列
    if price_column not in data.columns:
        raise ValueError(f"数据集必须包含 '{price_column}' 列。")

    if time_column not in data.columns:
        raise ValueError(f"数据集必须包含 '{time_column}' 列。")

    # 计算每日涨跌幅
    data['Change'] = data[price_column].pct_change()

    # 初始化存储结果的列表
    avg_changes = []

    # 遍历每个时间点
    for T in range(len(data)):
        # 确定后 n 单位时间的索引范围
        start_idx = T + 1
        end_idx = T + n + 1

        # 如果范围超出数据长度，截断到数据末尾
        changes_in_window = data['Change'][start_idx:end_idx]

        # 计算窗口内的平均涨跌幅
        avg_changes.append(changes_in_window.mean())

    # 将结果添加到数据集
    data[f'next_{n}_period_avg_change'] = avg_changes

    # 计算胜率
    win_rate = (data['Change'] > 0).mean()

    # 返回结果
    return data, win_rate

data_folder = os.path.join("..", "data", "binance", "BTCUSDT", "15m")
dataset = load_dataset_from_folder(data_folder)
result_data, win_rate = calculate_avg_change_next_n_periods(dataset, 3)
print(result_data)
print(win_rate)
