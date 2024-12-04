from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from get_btc_info import get_btc_data

start_date = datetime(2023, 11, 1)
end_date = datetime(2024, 11, 1)
timeframe = '1m'
crypto = "BTCUSDT"
exchange_name1 = 'bybit'
exchange_name2 = 'okx'

binance_data = get_btc_data(start_date, end_date, timeframe, exchange_name1, crypto)
bybit_data = get_btc_data(start_date, end_date, timeframe, exchange_name2, crypto)

def compare_dataframes(df1, df2):
    """
    比较两个 DataFrame 中相同时间戳的 open、close、high、low 差值。

    :param df1: 第一个 DataFrame，必须包含 'time', 'open', 'close', 'high', 'low' 列。
    :param df2: 第二个 DataFrame，必须包含 'time', 'open', 'close', 'high', 'low' 列。
    :return: 包含差值的 DataFrame，其中包括 'time', 'open_diff', 'close_diff', 'high_diff', 'low_diff' 列。
    """
    # 确保 'time' 列为 datetime 格式
    df1['time'] = pd.to_datetime(df1['time'])
    df2['time'] = pd.to_datetime(df2['time'])

    # 合并两个 DataFrame，按 'time' 对齐
    merged = pd.merge(df1, df2, on='time', suffixes=('_df1', '_df2'))
    print(merged.columns)
    # 计算差值
    merged['open_diff'] = merged['open_df1'] - merged['open_df2']
    merged['close_diff'] = merged['close_df1'] - merged['close_df2']
    merged['high_diff'] = merged['high_df1'] - merged['high_df2']
    merged['low_diff'] = merged['low_df1'] - merged['low_df2']
    # 计算每列绝对值大于 60 的出现次数
    threshold = 150
    open_diff_count = (merged['open_diff'].abs() > threshold).sum()
    close_diff_count = (merged['close_diff'].abs() > threshold).sum()
    high_diff_count = (merged['high_diff'].abs() > threshold).sum()
    low_diff_count = (merged['low_diff'].abs() > threshold).sum()

    # 输出结果
    print(f"Absolute value > {threshold} occurrences:")
    print(f"open_diff: {open_diff_count}")
    print(f"close_diff: {close_diff_count}")
    print(f"high_diff: {high_diff_count}")
    print(f"low_diff: {low_diff_count}")
    # 返回包含时间和差值的 DataFrame
    return merged[['time', 'open_diff', 'close_diff', 'high_diff', 'low_diff','open_df1']]


def plot_diff_distribution(diff_df):
    """
    绘制差值分布图，展示 open_diff、close_diff、high_diff、low_diff 的分布情况。

    :param diff_df: 包含差值的 DataFrame，必须有 'open_diff', 'close_diff', 'high_diff', 'low_diff' 列。
    """
    # 设置图形大小
    plt.figure(figsize=(12, 8))

    # 绘制每种差值的分布图
    for col in ['open_diff', 'close_diff', 'high_diff', 'low_diff']:
        plt.hist(diff_df[col], bins=50, alpha=0.6, label=col)

    # 图形美化
    plt.title("Distribution of Differences (Diff)", fontsize=16)
    plt.xlabel("Difference Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # 显示图形
    plt.show()



# 示例调用
result = compare_dataframes(binance_data, bybit_data)
plot_diff_distribution(result)
print(result)
filtered_result = result[abs(result["open_diff"]) > result["open_df1"] * 0.002]
pd.set_option('display.max_columns', None)
print(filtered_result)