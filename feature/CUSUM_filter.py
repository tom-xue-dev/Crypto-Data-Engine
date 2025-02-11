import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from feature_generation import alpha1, alpha6, alpha8, alpha10, alpha19, alpha20, alpha24, alpha25, alpha26, alpha32, \
    alpha35, alpha44, alpha46, alpha49, alpha51, alpha68, alpha84, alpha94, alpha95, alpha2,alpha102
import utils as u


def triple_barrier_labeling(prices, upper_pct=0.03, lower_pct=0.03, max_time=30):
    """
    采用 NumPy 向量化方式计算 Triple Barrier Labeling，提高计算效率。

    参数:
    prices: pd.Series, 价格数据
    upper_pct: float, 上界百分比 (默认 3%)
    lower_pct: float, 下界百分比 (默认 3%)
    max_time: int, 最长持有期 (默认 10 天)

    返回:
    labels: pd.Series, -1 (下限触发), 0 (时间到期), 1 (上限触发)
    """
    price_idx = prices.index
    prices = prices.to_numpy()  # 转换为 NumPy 数组，提高计算效率
    n = len(prices)
    labels = np.zeros(n, dtype=int)  # 预填充 0
    upper_barrier = prices * (1 + upper_pct)
    lower_barrier = prices * (1 - lower_pct)

    for t in range(n - max_time):
        future_prices = prices[t + 1: t + max_time + 1]

        # 找到第一个触碰上/下界的位置
        hit_upper = np.where(future_prices >= upper_barrier[t])[0]
        hit_lower = np.where(future_prices <= lower_barrier[t])[0]

        if hit_upper.size > 0 and (hit_lower.size == 0 or hit_upper[0] < hit_lower[0]):
            labels[t] = 1  # 触碰上界
        elif hit_lower.size > 0 and (hit_upper.size == 0 or hit_lower[0] < hit_upper[0]):
            labels[t] = -1  # 触碰下界
        else:
            labels[t] = 0  # 仅时间屏障触发

    return pd.Series(labels, index=price_idx)

def symmetric_cusum_filter(returns, threshold):
    """
    对输入的收益率序列应用对称CUSUM滤波器，仅返回事件的时间戳列表。
    """
    events = []
    s_pos, s_neg = 0, 0  # 初始化正负累计和
    for t, r in returns.items():
        s_pos = max(0, s_pos + r)
        s_neg = min(0, s_neg + r)

        if s_pos > threshold:
            events.append(t)  # 仅添加时间戳
            s_pos = 0  # 触发事件后重置
        elif s_neg < -threshold:
            events.append(t)  # 仅添加时间戳
            s_neg = 0  # 触发事件后重置

    return events


def generate_filter_df(data, sample_column='close', threshold=5 * 0.007):
    """
    对dataset进行cusum 采样
    params:
    data:要采样的dataset，必须包含 sample_column 列
    sample_column:采样的列名，通常为开盘收盘最高最低
    threshold:采样阈值
    """
    data['log_close'] = np.log(data[sample_column])
    data['returns'] = data.groupby('asset')['log_close'].diff()
    asset_events = {}
    for asset, group in data.groupby('asset'):
        events = symmetric_cusum_filter(group['returns'].dropna(), threshold)
        asset_events[asset] = events

    # 4. 利用事件时间点从原始数据中采样（提取整行 OHLC 数据）
    sampled_list = []
    for asset, events in asset_events.items():
        events_for_asset = [t for (t, a) in events if a == asset]

        # 提取 asset 对应的数据：
        asset_data = data.xs(asset, level='asset')
        sampled_asset = asset_data.loc[asset_data.index.isin(events_for_asset)]

        # 如果需要，可以给结果添加资产标签
        sampled_asset = sampled_asset.copy()
        sampled_asset['asset'] = asset
        sampled_list.append(sampled_asset)

    # 合并所有资产的采样数据
    sampled_data = pd.concat(sampled_list)
    sampled_data = sampled_data.sort_index()

    filter_data = sampled_data.set_index('asset', append=True)
    return filter_data


if __name__ == '__main__':
    start = "2020-1-1"
    end = "2024-12-31"
    assets = select_assets(start_time=start, spot=True, n=20)
    # print(assets)
    # assets = ['BTC-USDT_spot']
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    #data['label'] = data.groupby(level='asset', group_keys=False)['close'].apply(triple_barrier_labeling)
    data['vwap'] = u.vwap(data)
    # 计算每个资产的对数收益率（当前与前一时点比较）
    #print(len(data))
    data = generate_filter_df(data, sample_column='close', threshold=5 * 0.007)
    # 计算每个 asset 对应的列数（特征数量）
    # 计算每个 asset 在 MultiIndex DataFrame 中的行数
    # print(data['label'].value_counts())

    data = data.dropna()


    # alpha_funcs = [
    #     ('alpha1', alpha1),
    #     ('alpha6', alpha6),
    #     ('alpha8', alpha8),
    #     ('alpha10', alpha10),
    #     ('alpha19', alpha19),
    #     ('alpha24', alpha24),
    #     ('alpha26', alpha26),
    #     ('alpha32', alpha32),
    #     ('alpha35', alpha35),
    #     ('alpha46', alpha46),
    # ]
    # #
    # for col_name, func in alpha_funcs:
    #     data[col_name] = func(data)

    # data['alpha8'] = alpha8(data)
    data['alpha'] = alpha102(data)

    # print("max is",np.max(data['alpha']))
    # print(data['alpha'])

    data = data.dropna()
    # train_dict = {
    #     'alpha1': data['alpha1'].values,
    #     'alpha6': data['alpha6'].values,
    #     'alpha8': data['alpha8'].values,
    #     'alpha10': data['alpha10'].values,
    #     'alpha19': data['alpha19'].values,
    #     'alpha24': data['alpha24'].values,
    #     'alpha26': data['alpha26'].values,
    #     'alpha32': data['alpha32'].values,
    #     'alpha35': data['alpha35'].values,
    #     'alpha46': data['alpha46'].values,
    #     'label': data['label'].values
    # }
    #
    # #
    # print(train_dict)
    #
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(train_dict, f)
    print(data)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    print(len(data))
    daily_ic = data.groupby('asset').apply(lambda x: x['alpha'].corr(x['future_return'], method='spearman'))
    print("IC:", daily_ic.mean())
    print("IR", daily_ic.mean() / daily_ic.std())
