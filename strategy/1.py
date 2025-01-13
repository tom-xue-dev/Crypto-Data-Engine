import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from read_large_files import load_filtered_data_as_list, select_assets
from scipy.stats import pearsonr

start = "2022-1-1"
end = "2022-11-30"
assets = select_assets(spot=True, n=10)

data = load_filtered_data_as_list(start, end, assets, level="15min")
print(assets)
data = pd.concat(data, ignore_index=True)
data = data.set_index(["time", "asset"])
slopes = []  # 保存每个窗口的斜率
future_returns = []  # 保存每个窗口的未来涨跌幅
window_size = 50
z_scores = []
avg_changes = []
for asset, df in data.groupby('asset'):
    for i in range(window_size, len(df) - window_size):
        x_window = df['low'].iloc[i - window_size:i].values  # 自变量 (low)
        y_window = df['high'].iloc[i - window_size:i].values  # 因变量 (high)
        pre_price = df['close'].iloc[i - window_size]
        current_price = df['close'].iloc[i + window_size]
        # 计算标准差
        std_x = np.std(x_window, ddof=1)  # std(x)
        std_y = np.std(y_window, ddof=1)  # std(y)

        # 计算相关系数 corr(y, x)
        corr_xy = np.corrcoef(x_window, y_window)[0, 1]
        if std_x < 1e-6:
            data.loc[df.index[i], 'signal'] = 0
            continue
        # 计算 zscore(β)
        beta = (std_y / std_x) * corr_xy  # 斜率 β
        zscore_beta = (std_y / std_x) * corr_xy * (corr_xy ** 2)  # zscore(β)
        avg_change = (current_price - pre_price) / pre_price
        avg_changes.append(avg_change)
        z_scores.append(zscore_beta)
        if zscore_beta > 1:
            data.loc[df.index[i], 'signal'] = 1
        else:
            data.loc[df.index[i], 'signal'] = 0
print(data)
with open("data_signal.pkl", 'wb') as f:
    pickle.dump(data, f)
data1 = pd.DataFrame({
    'zscore': z_scores,
    'avg_change': avg_changes
})
bins = np.linspace(-0.1, 2, 20)  # 将 zscore 划分为 5 个区间
data1['zscore_bin'] = pd.cut(data1['zscore'], bins=bins)

# 按区间分组并过滤样本量大于一定数量的区间
min_samples = 100  # 设置最小样本数量
grouped = data1.groupby('zscore_bin').filter(lambda x: len(x) >= min_samples)

# 统计区间的平均涨跌幅和样本量
result = grouped.groupby('zscore_bin').agg(
    avg_change_mean=('avg_change', 'mean'),  # 计算平均涨跌幅
    sample_count=('avg_change', 'count')  # 计算样本数量
).reset_index()

print(result)
