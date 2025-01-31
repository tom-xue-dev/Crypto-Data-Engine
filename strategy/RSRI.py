import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from read_large_files import load_filtered_data_as_list, select_assets,map_and_load_pkl_files
from scipy.stats import pearsonr


start = "2019-1-1"
end = "2022-12-30"
assets = select_assets(spot=True, n=10)
#assets = ['BTC-USDT_spot']
data = map_and_load_pkl_files(start_time=start, end_time=end, asset_list=assets, level='15min')
print(data)


slopes = []  # 保存每个窗口的斜率
future_returns = []  # 保存每个窗口的未来涨跌幅 (150,300,150)
z_scores = []
avg_changes = []
beta_values = []
signals = []
rs_record = []
window_size = 200
M = 600
threshold = 2.0
data['signal'] = 0

for asset, df in data.groupby('asset'):
    for i in range(0, len(df)):
        # 提取滑动窗口数据
        if i < window_size or i + 50 >= len(df):
            beta_values.append(np.nan)
            continue
        x_window = df['low'].iloc[i - window_size:i].values  # 自变量
        y_window = df['high'].iloc[i - window_size:i].values  # 因变量
        current_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + 50]
        sigma_low = np.std(y_window, ddof=1)
        sigma_high = np.std(x_window, ddof=1)
        X = sm.add_constant(x_window)
        # 创建 OLS 模型并拟合
        model = sm.OLS(y_window, X)
        results = model.fit()
        if np.all(x_window == x_window[0]):
            print("自变量是常数列，无法进行回归")
            print(x_window)
            print(y_window)
            beta_values.append(np.nan)
        else:
            beta = results.params[1]  # 斜率
            beta_values.append(beta)
        rs = results.rsquared
        rs_record.append(rs)
        if len(beta_values) >= M:
            mean = np.nanmean(beta_values[-M:])
            std = np.nanstd(beta_values[-M:])

    df['return_5'] = df['close'].shift(-5).rolling(window=5).mean() / df['close']
    df['return_10'] = df['close'].shift(-10).rolling(window=10).mean() / df['close']
    df['beta'] = beta_values
    # 计算每个未来收益的 IC
    ic_5 = df[['beta', 'return_5']].dropna().corr(method='spearman').iloc[0, 1]
    ic_10 = df[['beta', 'return_10']].dropna().corr(method='spearman').iloc[0, 1]
    # ic_values = []
    # ic_values.append(ic_5)
    # # 计算 IC 的均值
    # # 计算 IC 的均值
    ic_mean = np.mean([ic_5, ic_10])
    ic_std = np.std([ic_5, ic_10])
    print(f"IC_5: {ic_5:.4f}")
    print(f"IC_10: {ic_10:.4f}")
    print(f"IC Mean: {ic_mean:.4f}")
    print(f"ICIR:{ic_mean / ic_std}")
# print(data[140:160])
# with open(f"data_signal({window_size},{M},{threshold}).pkl", 'wb') as f:
#     pickle.dump(data, f)
# data1 = pd.DataFrame({
#     'zscore': signals,
#     'avg_change': avg_changes
# })
#
# bins = np.linspace(-2, 2, 40)
# data1['zscore_bin'] = pd.cut(data1['zscore'], bins=bins)
#
# # 按区间分组并过滤样本量大于一定数量的区间
# min_samples = 10  # 设置最小样本数量
# grouped = data1.groupby('zscore_bin').filter(lambda x: len(x) >= min_samples)
#
# correlation = data1['zscore'].corr(data1['avg_change'])
# print("相关系数:", correlation)
#
# # 统计区间的平均涨跌幅和样本量
# result = grouped.groupby('zscore_bin').agg(
#     avg_change_mean=('avg_change', 'mean'),  # 平均涨跌幅
#     sample_count=('avg_change', 'count'),  # 样本数量
#     win_rate=('avg_change', lambda x: (x > 0).mean())  # 胜率
# ).reset_index()
#
# print(result)
