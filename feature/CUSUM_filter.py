import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from feature_generation import alpha1, alpha6, alpha8, alpha10, alpha19, alpha20, alpha24, alpha25, alpha26, alpha32, \
    alpha35, alpha44, alpha46, alpha49, alpha51, alpha68, alpha84, alpha94, alpha95, alpha2
import utils as u


def symmetric_cusum_filter(returns, threshold):
    """
    对输入的收益率序列应用对称CUSUM滤波器，仅返回事件的时间戳列表。
    """
    events = []
    s_pos, s_neg = 0, 0  # 初始化正负累计和
    print(returns)
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
    start = "2022-1-1"
    end = "2024-12-31"
    assets = select_assets(start_time=start, spot=True, n=50)
    # print(assets)
    # assets = ['BTC-USDT_spot']
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)

    data['vwap'] = u.vwap(data)
    # 计算每个资产的对数收益率（当前与前一时点比较）
    print(len(data))
    data = generate_filter_df(data, sample_column='close', threshold=5 * 0.007)
    # 计算每个 asset 对应的列数（特征数量）
    # 计算每个 asset 在 MultiIndex DataFrame 中的行数
    row_counts = data.groupby(level="asset").size()

    # 输出结果
    print(row_counts)

    data = data.dropna()

    #
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
    #     ('alpha49', alpha49),
    # ]
    # #
    # for col_name, func in alpha_funcs:
    #     data[col_name] = func(data)
    #
    # # data['alpha8'] = alpha8(data)
    # # data['alpha'] = alpha84(data)
    # data = data.dropna()
    # # print("max is",np.max(data['alpha']))
    # # print(data['alpha'])
    # data['label'] = np.where(data['future_return'] > 0, 1, 0)
    #
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
    #     'alpha49': data['alpha49'].values,
    #     'label': data['label'].values
    # }
    # #
    # # #
    # # print(train_dict)
    # #
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(train_dict, f)
    # print(data)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    print(len(data))
    daily_ic = data.groupby('asset').apply(lambda x: x['alpha1'].corr(x['future_return'], method='spearman'))
    print("IC:", daily_ic.mean())
    print("IR", daily_ic.mean() / daily_ic.std())
