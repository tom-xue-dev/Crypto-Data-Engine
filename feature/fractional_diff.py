import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as u
from feature.RSRI import compute_alpha_parallel
from feature.read_large_files import select_assets, map_and_load_pkl_files
from Factor import alpha1, alpha2, alpha25, alpha32, alpha46, alpha51, alpha95, alpha9, alpha103, alpha35, alpha101, \
    alpha104, alpha106, alpha105, alpha102, alpha107, alpha108
from IC_calculator import compute_zscore, compute_ic
from statsmodels.tsa.stattools import adfuller

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


if __name__ == '__main__':
    start = "2020-1-1"
    end = "2020-12-31"
    assets = select_assets(start_time=start, spot=True, m=20)
    print(assets)
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    d = 0.2
    data['returns'] = u.returns(data)
    #origin = data['close'].values
    data['log_close'] = np.log(data['close'])
    data['vwap'] = u.vwap(data)
    columns = ['open', 'high', 'low', 'close', 'volume']
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    for column in columns:
        data[column] = np.log(data[column])
        data[column] = data[column].groupby('asset').apply(lambda x: fracdiff(x, d)).droplevel(0)

    alpha_funcs = [
        ('alpha1', alpha1),
        ('alpha2', alpha2),
        ('alpha9', alpha9),
        ('alpha25', alpha25),
        ('alpha32', alpha32),
        ('alpha46', alpha46),
        ('alpha95', alpha95),
        ('alpha101', alpha101),
        ('alpha102', alpha102),
        ('alpha103', alpha103),
        ('alpha104', alpha104),
        ('alpha105', alpha105),
        ('alpha106', alpha106),
        ('alpha107', alpha107),
        ('alpha108', alpha108)
    ]
    for name, func in alpha_funcs:
        print(f"=== Now computing {name} in parallel... ===")
        data = compute_alpha_parallel(data, alpha_func=func, n_jobs=16)
    for column in data.columns:
        if column not in ['open', 'close', 'high', 'low', 'APO', 'RSI', 'returns', 'volume', 'return', 'log_close',
                          'future_return', 'vwap', 'amount', 'beta', 'downsidevolatility', 'upsidevolatility',
                          'volumestd']:
            ic = compute_ic(df=data, feature_column=column, return_column='future_return')
            print(column, "IC_MEAN:", np.mean(ic), "IR", np.mean(ic) / np.std(ic))
    # # 绘图对比
    # print(origin[100:])
    # print(data['close'].values[100:])
    # corr_matrix = np.corrcoef(origin[100:],data['close'].values[100:])
    # print(corr_matrix[0,1])
    # 计算 ADF 检验
    # adf_result = adfuller(np.cumsum(data['close'].values))
    #
    # # 解析输出结果
    # print("ADF Statistic:", adf_result[0])  # ADF 统计量
    # print("p-value:", adf_result[1])  # p 值
    # print("Critical Values:", adf_result[4])  # 1%、5%、10% 显著性水平的临界值

    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index.get_level_values('time'), origin, label='original series')
    # plt.plot(data.index.get_level_values('time'), data['close'], label=f'series after diff (d={d})', color='red')
    # plt.xlabel('time')
    # plt.ylabel('value')
    # plt.title('example')
    # plt.legend()
    # plt.show()
