import sys

import numpy as np
import pandas as pd

from read_large_files import load_filtered_data_as_list, select_assets, map_and_load_pkl_files
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def llt_filter(price, alpha):
    """
    使用 LLT 滤波器计算低延迟趋势线。
    :param price: 输入的价格序列 (list, np.array, 或 pd.Series)
    :param alpha: 平滑系数，范围 (0, 1)
    :return: 滤波后的 LLT 序列 (np.array)
    """
    # 强制转换为 NumPy 数组（兼容 Pandas Series）
    price = np.asarray(price)

    if len(price) < 3:
        raise ValueError("数据长度必须至少为 3")

    llt = np.zeros_like(price)
    llt[0] = price[0]
    llt[1] = price[1]

    c0 = alpha - (alpha ** 2) / 4
    c1 = (alpha ** 2) / 2
    c2 = alpha - (3 * alpha ** 2) / 4
    d1 = 2 * (1 - alpha)
    d2 = (1 - alpha) ** 2

    for t in range(2, len(price)):
        llt[t] = (
                c0 * price[t] +
                c1 * price[t - 1] -
                c2 * price[t - 2] +
                d1 * llt[t - 1] -
                d2 * llt[t - 2]
        )

    return llt


while True:
    start = "2019-1-1"
    end = "2022-12-30"
    assets = ['BTC-USDT_spot']
    a = assets.copy()
    data = map_and_load_pkl_files(start_time=start, end_time=end, asset_list=assets, level='15min')
    # 示例数据
    if data.empty:
        continue
    print(a)
    data = data['close']
    alpha = 0.05
    llt_data = llt_filter(data, alpha)

    k = 30  # 差分的周期(例如30根15分钟K)
    # forward_diff_k = (future - now)/future的写法可根据你自己的定义调整
    # 这里示例用 (data.shift(-k) - data)/data.shift(-k)
    llt_data = pd.Series(llt_data, index=data.index)

    forward_diff_k = (llt_data - llt_data.shift(k)) / llt_data

    factor_raw = forward_diff_k.dropna()

    factor_smooth = factor_raw.ewm(alpha=0.3).mean().dropna()
    # factor_smooth = factor_raw

    # factor_smooth 就是我们最终想要用来评估的因子值

    # ============== 4. (可选) 根据阈值生成多空信号 ================
    # 如果你需要一个离散的多空信号，可以这样做：
    # # 定义滚动窗口参数
    # window_size = 3000  # 窗口长度（如60根K线）
    # min_periods = 30  # 最小有效数据量（可设为1，但需谨慎）
    #
    # # 计算滚动均值和标准差
    # rolling_mean = factor_smooth.rolling(window=window_size, min_periods=min_periods).mean()
    # rolling_std = factor_smooth.rolling(window=window_size, min_periods=min_periods).std()
    #
    # # 动态阈值
    # upper_thresh = rolling_mean + 1 * rolling_std
    # lower_thresh = rolling_mean - 1 * rolling_std
    #
    # # 处理初始 NaN（示例：后向填充）
    # upper_thresh = upper_thresh.fillna(method='bfill')
    # lower_thresh = lower_thresh.fillna(method='bfill')
    #
    # # 生成信号
    # signals = pd.Series(0, index=factor_smooth.index)
    # signals[factor_smooth > upper_thresh] = 1
    # signals[factor_smooth < lower_thresh] = -1
    #
    # # 删除剩余的 NaN（如果有）
    # signals = signals.dropna()
    # signals 即根据阈值得到的多空信号
    # 后面若只想看因子与未来收益的IC，可忽略signals，直接用 factor_smooth

    # ============== 5. 计算未来5、10、20根K线的收益率 ==============
    # 未来 n 根 K 线的简单收益率 = (未来价格 - 当前价格) / 当前价格
    # 用 shift(-n) 来代表向后 n 根
    future_ret_5 = data.shift(-5) / data - 1
    future_ret_10 = data.shift(-10) / data - 1
    future_ret_20 = data.shift(-20) / data - 1
    future_ret_30 = data.shift(-30) / data - 1
    # 对齐索引，去掉 NaN
    common_idx_5 = factor_smooth.index.intersection(future_ret_5.dropna().index)
    common_idx_10 = factor_smooth.index.intersection(future_ret_10.dropna().index)
    common_idx_20 = factor_smooth.index.intersection(future_ret_20.dropna().index)
    common_idx_30 = factor_smooth.index.intersection(future_ret_30.dropna().index)

    # ============== 6. 计算因子与未来收益的IC(相关系数) =============
    ic_5 = factor_smooth.loc[common_idx_5].corr(future_ret_5.loc[common_idx_5])
    ic_10 = factor_smooth.loc[common_idx_10].corr(future_ret_10.loc[common_idx_10])
    ic_20 = factor_smooth.loc[common_idx_20].corr(future_ret_20.loc[common_idx_20])
    ic_30 = factor_smooth.loc[common_idx_20].corr(future_ret_30.loc[common_idx_30])
    print("IC for future 5 bars :", ic_5)
    print("IC for future 10 bars:", ic_10)
    print("IC for future 20 bars:", ic_20)
    print("IC for future 30 bars:", ic_30)
#
# # ============== (可选) 若要看离散多空信号与未来收益的相关性 =================
# # 若你想看 signals（离散多空）与未来收益的相关性，则可以：
# ic_5_signal = signals.loc[common_idx_5].corr(future_ret_5.loc[common_idx_5])
# ic_10_signal = signals.loc[common_idx_10].corr(future_ret_10.loc[common_idx_10])
# ic_20_signal = signals.loc[common_idx_20].corr(future_ret_20.loc[common_idx_20])
#
# print("Signal IC for 5 bars :", ic_5_signal)
# print("Signal IC for 10 bars:", ic_10_signal)
# print("Signal IC for 20 bars:", ic_20_signal)
