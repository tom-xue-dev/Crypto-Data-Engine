import sys

import numpy as np
import pandas as pd

from read_large_files import load_filtered_data_as_list, select_assets, map_and_load_pkl_files
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

start = "2021-1-1"
end = "2022-12-30"
assets = select_assets(spot=True, n=5)
# assets = ['BTC-USDT_spot','ETH-USDT_spot']

df = map_and_load_pkl_files(start_time=start, end_time=end, asset_list=assets, level='15min')
print(df)
# 重置索引并提取日期
df = df.reset_index()
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date  # 提取日期列用于分组
df = df.set_index(['time', 'asset'])
df['return'] = df.groupby('asset')['close'].pct_change()

# 删除缺失值（首行无收益率）
df = df.dropna(subset=['return'])

# 定义滚动窗口大小（例如：过去20个15分钟窗口）
window_size = 20


# 按日期分组，计算每个交易日内滚动自相关性
def calculate_autocorr(group):
    # 滚动计算滞后1期的自相关性
    group['autocorr_factor'] = group['return'].rolling(window=window_size).apply(lambda x: x.autocorr(lag=1))
    return group


# 应用分组计算
df = df.groupby(['date', 'asset'], group_keys=False).apply(calculate_autocorr, include_groups=False)

# 计算下一期的收益率（未来1个15分钟窗口）
df['future_return'] = df.groupby('asset')['close'].shift(-10).pct_change(fill_method=None)

# 删除未来收益率的缺失值
df = df.dropna(subset=['future_return'])

# 按时间点计算IC
ic_values = []
for timestamp, group in df.groupby('time'):
    if len(group) >= 2:  # 至少需要2个数据点计算相关系数
        factor_rank = group['autocorr_factor'].rank()
        future_return_rank = group['future_return'].rank()
        ic, _ = spearmanr(factor_rank, future_return_rank)
        ic_values.append(ic)

# 计算IC均值
if len(ic_values) == 0:
    print("警告：无有效IC值，请检查数据或时间窗口设置！")
    ic_mean = np.nan  # 或其他默认值
else:
    ic_mean = np.nanmean(ic_values)
print(f"自相关性因子的IC均值为: {ic_mean:.4f}")
