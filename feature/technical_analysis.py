import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import talib

def func(group):
    """
    对单个资产（group DataFrame）进行滚动窗口计算，
    对于每个窗口计算 trend = (当前收盘价 - 窗口低点) / (窗口高点 - 窗口低点)。
    返回的 Series 必须与 group.index 对齐。
    """
    window = 30
    trend_values = []
    for i in range(len(group)):
        # 如果不足 window 个数据，则返回 NaN
        if i < window:
            trend_values.append(np.nan)
        else:
            # 取当前行及之前 window-1 行构成窗口
            window_df = group.iloc[i - window: i]
            max_high = window_df['high'].max()  # 计算最高值

            count = (abs(window_df['high'] - max_high) < max_high * 0.01).sum()  # 统计满足条件的个数
            diff = np.diff(window_df['low'].values)
            n_pos = np.sum(diff > 0)
            n_neg = np.sum(diff < 0)
            if abs(group.iloc[i]['high'] - max_high) < max_high * 0.02 and n_pos / n_neg > 1.5:
                trend_values.append(-1)
            else:
                trend_values.append(0)

            # min_low = window_df['low'].min()
            # current_close = group.iloc[i]['close']
            # if current_close > max_high:
            #     trend_values.append(1)
            # elif current_close < min_low:
            #     trend_values.append(-1)
            # else:
            #     trend_values.append(0)
    return pd.Series(trend_values, index=group.index)


def parallel_apply(df, func, group_key):
    """
    利用多进程对 DataFrame 按 group_key 分组后执行 func 计算。
    """
    # 先按 group_key 分组，构造分组列表（注意：此处 group 包含原来的索引信息）
    groups = [group for _, group in df.groupby(group_key)]

    with ProcessPoolExecutor() as executor:
        # executor.map 会并行地将每个 group 传给 func 函数
        results = list(executor.map(func, groups))

    # 将所有计算结果合并回一个 Series，并按索引排序以对齐原 DataFrame
    trend_series = pd.concat(results).sort_index()
    return trend_series


def trend_analysis(df):
    trend_series = parallel_apply(df, func, 'asset')

    # 如果需要把结果赋值回原 DataFrame，可以直接：
    df['trend'] = trend_series

    return df['trend']



