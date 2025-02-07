import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    """
    对 pandas.Series 类型的时间序列进行分数差分处理。

    参数:
      - series: 输入的时间序列 (pd.Series)
      - d: 分数差分阶数 (可以为非整数)
      - window: 固定的窗口长度

    返回:
      - 分数差分后的序列 (pd.Series)，序列前部因数据不足被填充为 NaN。
    """
    weights = fracdiff_weights(d, window)
    width = len(weights) - 1  # 需要的滞后期数
    result = [np.nan] * len(series)
    for i in range(width, len(series)):
        window_data = series.iloc[i - width: i + 1].values
        result[i] = np.dot(weights, window_data)
    return pd.Series(result, index=series.index)

# --------------------------
# 示例：生成随机游走序列并进行分数差分
# --------------------------

# 设置随机种子以便复现结果
# np.random.seed(0)
#
# # 生成随机游走序列（非平稳序列）
# T = 1000
# # 累积和模拟随机游走
# ts = pd.Series(np.cumsum(np.random.normal(0, 1, T)), name='Random Walk')
#
# # 设置分数差分阶数
# d = 0.4
# ts_fracdiff = fracdiff(ts, d)
#
# # 绘图对比
# plt.figure(figsize=(12, 6))
# plt.plot(ts.index, ts, label='original series')
# plt.plot(ts_fracdiff.index, ts_fracdiff, label=f'series after diff (d={d})', color='red')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.title('example')
# plt.legend()
# plt.show()
