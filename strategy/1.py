import pickle
import sys
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, select_assets
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from strategy import DualMAStrategy

from analysis import process_future_performance_in_pool


def calculate_garman_klass_volatility(group, window):
    """
    在 DataFrame 中添加 Garman-Klass 波动率列。
    """
    group['GK_vol'] = (
            0.5 * (np.log(group['high'] / group['low'])) ** 2 -
            (2 * np.log(2) - 1) / window * (np.log(group['close'] / group['open'])) ** 2
    )
    group['GK_vol_rolling'] = group['GK_vol'].rolling(window=window).mean()
    return group


def process_asset_signals_wrapper(params):
    group, window, threshold, volatility_threshold = params
    return process_asset_group(group, window, threshold, volatility_threshold)


def process_asset_group(group, window, threshold, volatility_threshold):
    """
    对单个资产组的数据进行信号生成。
    """

    for i in range(len(group)):
        if i - int(window / 50) < window * 5:
            continue  # 跳过窗口不足的前几天
        if group.iloc[i]["signal"] == 1:

            if not group.iloc[i]["MA50"] > group.iloc[i]["MA2400"]:
                group.loc[group.index[i], "signal"] = 0
                continue
            if not group.iloc[i-200]["MA50"] > group.iloc[i]["MA50"]:
                group.loc[group.index[i], "signal"] = 0
                continue
            if not group.iloc[i-400]["MA100"] > group.iloc[i]["MA100"]:
                group.loc[group.index[i], "signal"] = 0
                continue

    return group


def generate_signal(data, window, threshold, volatility_threshold=None):
    """
    生成交易信号列，基于多线程处理。
    """
    # 按资产分组
    grouped_data = [group for _, group in data.groupby("asset")]

    params = [(group, window, threshold, volatility_threshold) for group in grouped_data]
    # 使用线程池并行处理每个资产组

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_asset_signals_wrapper, params)

    result_df = pd.concat(results)
    # 可选：按索引排序以恢复原始顺序
    result_df = result_df.sort_index()

    return result_df


if __name__ == "__main__":
    # 从 .pkl 文件中加载数据

    with open("data.pkl", "rb") as file:
        data = pickle.load(file)
    print(data.iloc[0])
    strategy = DualMAStrategy(dataset=data, long_period=100, short_period=200)
    print("start generate 100")
    data = strategy.dataset
    print("start generate 50")
    strategy_2 = DualMAStrategy(dataset=data, long_period=50, short_period=2400)
    # print(data[:len(data) // 4])

    data = strategy_2.dataset
    data = generate_signal(data, 1200, 0.01)
    n_splits = 24
    split_size = len(data) // n_splits  # 每等分的大小
    # 循环处理每个部分
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else len(data)  # 确保最后一部分包括所有剩余行
        subset = data[start_idx:end_idx]  # 当前部分
        n_days_list = list(range(5, 600, 20))
        print(subset.iloc[0])
        results = process_future_performance_in_pool(subset, n_days_list, signal=1)
        for n_days, avg_return, prob_gain, count in results:
            print(f"未来{n_days}根k线上涨概率为{prob_gain},上涨幅度{avg_return},总数{count}")
        print()
    #

    # 输出结果

