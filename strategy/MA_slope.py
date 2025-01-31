import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from read_large_files import load_filtered_data_as_list, select_assets, map_and_load_pkl_files
from cumsum_filter import getTEvents

# 示例收盘价数据
start = "2019-1-1"
end = "2022-12-30"
#assets = select_assets(spot=True, n=1)
assets = ['RARE-USDT_spot']
print(assets)
df = map_and_load_pkl_files(start_time=start, end_time=end, asset_list=assets, level='15min')

idx = getTEvents(df['close'], df.iloc[0]['close'] / 200)


print(len(idx), len(df))
df = df.loc[idx]

# 参数
n = 10  # n日均线窗口
k = 200  # 未来收益天数

# 计算n日均线
df["n_day_ma"] = df['close'].ewm(span=n, adjust=False).mean()
# df["n_day_ma"] = df['close'].rolling(n).mean()
# 计算均线斜率（简单使用两点斜率）
df["slope"] = df["n_day_ma"].diff() / 1  # 计算相邻点间的斜率
df["slope_normalized"] = (df["slope"] - df["slope"].rolling(k).mean()) / df["slope"].rolling(k).std()

# 计算未来k天收益
df["future_return"] = df["close"].shift(-k) / df["close"] - 1
df = df.dropna()

for asset, group in df.groupby('asset'):
    group['return_5'] = group['close'].shift(-5).rolling(window=5).mean() / group['close']
    group['return_10'] = group['close'].shift(-10).rolling(window=10).mean() / group['close']
    group['return_20'] = group['close'].shift(-20).rolling(window=20).mean() / group['close']
    group['return_50'] = group['close'].shift(-20).rolling(window=50).mean() / group['close']
    # 计算每个未来收益的 IC
    ic_5 = group[['slope_normalized', 'return_5']].dropna().corr(method='spearman').iloc[0, 1]
    ic_10 = group[['slope_normalized', 'return_10']].dropna().corr(method='spearman').iloc[0, 1]
    ic_20 = group[['slope_normalized', 'return_20']].dropna().corr(method='spearman').iloc[0, 1]
    ic_50 = group[['slope_normalized', 'return_50']].dropna().corr(method='spearman').iloc[0, 1]
    # # 计算 IC 的均值
    ic_mean = np.mean([ic_5, ic_10, ic_20])
    ic_std = np.std([ic_5, ic_10, ic_20])
    print(f"IC_5: {ic_5:.4f}")
    print(f"IC_10: {ic_10:.4f}")
    print(f"IC_20: {ic_20:.4f}")
    print(f"IC Mean: {ic_mean:.4f}")

    print(f"ICIR:{ic_mean / ic_std}")
