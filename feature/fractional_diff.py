import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as u
from Dataloader import DataLoader,DataLoaderConfig
from Factor import *
from IC_calculator import compute_zscore, compute_ic
from statsmodels.tsa.stattools import adfuller
from scipy.signal import lfilter
def fracdiff_weights(d, window=30):
    """
    计算分数差分的权重，使用固定窗口长度。
    返回的权重按时间顺序排列，最后一个权重对应最新的观测值。
    """
    w = [1.0]
    j = 1
    while len(w) < window:
        w_j = -w[-1] * (d - j + 1) / j
        w.append(w_j)
        j += 1
    return np.array(w[::-1])


def fracdiff(series, d, window=30):
    w = fracdiff_weights(d, window)
    # 使用 lfilter 做 FIR 滤波（加权和）
    filtered = lfilter(w, [1], series.fillna(0).values)
    filtered[:window-1] = np.nan  # 前面无效区域设为NaN
    return pd.Series(filtered, index=series.index)

def fracdiff_implementation(df, columns=None, d=0.4, window=30):
    """
    """
    if columns is None:
        columns = ['open', 'high', 'low', 'close']
    result_list = []
    for asset, group in df.groupby('asset'):
        group_fd = group.copy()
        for col in columns:
            if col in group.columns:
                fd_series = fracdiff(group[col], d=d, window=window)
                group_fd[col] = fd_series
        result_list.append(group_fd)
    # 合并所有 asset
    result = pd.concat(result_list).sort_index()
    return result

