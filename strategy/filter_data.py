import pickle
import sys
import ast
import statsmodels.api as sm
import numpy as np
import pandas as pd

from read_large_files import map_and_load_pkl_files

start = "2020-11-1"
end = "2022-11-30"

# assets = select_assets(spot=True, n=1)

# print(assets)
assets = ["BTC-USDT_spot"]
data = map_and_load_pkl_files(start_time=start, end_time=end, level='1d', asset_list=assets)
#
with open("data.pkl", "rb") as f:
    data_day = pickle.load(f)
print(data)
window = 90
win = 30
correlation_coefficient = []
volumes = []
signals = []
avg_changes = []
# data['signal'] = 0
for asset, df in data.groupby('asset'):
    for i in range(0, len(df)):
        # 提取滑动窗口数据
        volumes.append(df['volume'].iloc[i])
        if i < win or i + 5 >= len(df):
            continue
        volume_mean = np.mean(volumes[i - win:i])
        if i > window + win:
            volume = df['volume'].iloc[i]
            signals.append(volume / volume_mean)
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + 3]
            avg_changes.append((future_price - current_price) / current_price)
            if data_day.loc[df.index[i], 'signal'] == 0:
                if volume / volume_mean > 1.3 or volume / volume_mean < 0.66:
                    data_day.loc[df.index[i], 'signal'] = -1

# print(data[140:160])
with open(f"data_signal_filter.pkl", 'wb') as f:
    pickle.dump(data_day, f)
data1 = pd.DataFrame({
    'zscore': signals,
    'avg_change': avg_changes
})

bins = np.linspace(0, 2, 10)
data1['zscore_bin'] = pd.cut(data1['zscore'], bins=bins)

# 按区间分组并过滤样本量大于一定数量的区间
min_samples = 10  # 设置最小样本数量
grouped = data1.groupby('zscore_bin').filter(lambda x: len(x) >= min_samples)

correlation = data1['zscore'].corr(data1['avg_change'])
print("相关系数:", correlation)

# 统计区间的平均涨跌幅和样本量
result = grouped.groupby('zscore_bin').agg(
    avg_change_mean=('avg_change', 'mean'),  # 平均涨跌幅
    sample_count=('avg_change', 'count'),  # 样本数量
    win_rate=('avg_change', lambda x: (x > 0).mean())  # 胜率
).reset_index()

print(result)
