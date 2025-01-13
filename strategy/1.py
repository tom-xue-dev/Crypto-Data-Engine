import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from read_large_files import load_filtered_data_as_list, select_assets
from scipy.stats import pearsonr

# start = "2021-10-1"
# end = "2022-12-30"

start = "2017-10-1"
end = "2020-12-30"
# assets = select_assets(spot=True, n=30)
assets = ['BTC-USDT_spot']

data = load_filtered_data_as_list(start, end, assets, level="1d")
# with open("data_BTC.pkl", "rb") as f:
#     data = pickle.load(f)
print(assets)
data = pd.concat(data, ignore_index=True)
data = data.set_index(["time", "asset"])
slopes = []  # 保存每个窗口的斜率
future_returns = []  # 保存每个窗口的未来涨跌幅
window_size = 20
M = 300
z_scores = []
avg_changes = []
beta_values = []
signals = []
for asset, df in data.groupby('asset'):
    for i in range(0, len(df)):
        # 提取滑动窗口数据
        if i < window_size or i +10 >= len(df):
            data.loc[df.index[i], 'signal'] = 0
            continue
        x_window = df['low'].iloc[i - window_size:i].values  # 自变量
        y_window = df['high'].iloc[i - window_size:i].values  # 因变量
        current_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + 10]

        sigma_low = np.std(y_window, ddof=1)  # 样本标准差
        sigma_high = np.std(x_window, ddof=1)
        rho = np.corrcoef(y_window, x_window)[0, 1]
        beta = rho * (sigma_high / sigma_low)
        # print(beta,beta1)
        beta_values.append(beta)
        data.loc[df.index[i], 'signal'] = 0
        # 如果已经累积了至少 M 个斜率值，则计算 z-score
        if len(beta_values) >= M:
            mean = np.mean(beta_values[-M:])
            std = np.std(beta_values[-M:])
            z_score = (beta - mean) / std * rho ** 2
            avg_change = (future_price - current_price) / current_price
            avg_changes.append(avg_change)
            signals.append(mean)
            if mean > 0.8:
                data.loc[df.index[i], 'signal'] = 1

#print(data)
with open("data_signal.pkl", 'wb') as f:
    pickle.dump(data, f)
data1 = pd.DataFrame({
    'zscore': signals,
    'avg_change': avg_changes
})
bins = np.linspace(0.718, 1, 20)
data1['zscore_bin'] = pd.cut(data1['zscore'], bins=bins)

# 按区间分组并过滤样本量大于一定数量的区间
min_samples = 10  # 设置最小样本数量
grouped = data1.groupby('zscore_bin').filter(lambda x: len(x) >= min_samples)

# 统计区间的平均涨跌幅和样本量
result = grouped.groupby('zscore_bin').agg(
    avg_change_mean=('avg_change', 'mean'),  # 平均涨跌幅
    sample_count=('avg_change', 'count'),  # 样本数量
    win_rate=('avg_change', lambda x: (x > 0).mean())  # 胜率
).reset_index()

print(result)
