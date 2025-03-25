import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from Factor import alpha104, alpha106, alpha1, alpha2, alpha9, alpha101, alpha102, alpha103, alpha107, alpha105, \
    alpha25, alpha32, alpha46, alpha95, alpha108
from concurrent.futures import ProcessPoolExecutor
import utils as u
from IC_calculator import compute_zscore
from feature_implementation import compute_alpha_parallel


def process_alpha_item(args):
    col_name, func, data = args
    # 这里对 data 调用 func 得到结果 Series
    result = func(data)
    return col_name, result


def symmetric_cusum_filter(returns: pd.Series, n: float, window: int = 20) -> list:
    """
    对输入的收益率序列应用对称 CUSUM 滤波器，仅返回事件的时间戳列表。
    阈值动态设置为过去 window 个收益率的标准差的 n 倍。

    参数：
        returns : pd.Series
            收益率序列，索引为时间戳。
        n : float
            阈值倍数，即阈值 = n * (过去 window 个收益率的标准差)。
        window : int, 默认 1200
            用于计算标准差的滚动窗口大小（过去 window 个数据点）。

    返回：
        list
            触发事件的时间戳列表。
    """
    events = []
    s_pos, s_neg = 0, 0  # 初始化正负累计和

    # 预先计算滚动标准差序列（阈值 = n * 标准差）
    # 注意：由于rolling默认包含当前数据点，为避免使用当前值，
    # 可以选择设置参数 closed='left'（pandas>=1.1.0），或者接受一定的滞后。
    thresholds = returns.rolling(window=window, min_periods=window, closed='left').std() * n

    for t, r in returns.items():
        s_pos = max(0, s_pos + r)
        s_neg = min(0, s_neg + r)

        current_threshold = thresholds.get(t, np.nan)
        # 如果当前阈值无法计算，则跳过
        if np.isnan(current_threshold):
            continue

        if s_pos > current_threshold:
            events.append(t)  # 记录事件时间戳
            s_pos = 0  # 重置正累计和
        elif s_neg < -current_threshold:
            events.append(t)  # 记录事件时间戳
            s_neg = 0  # 重置负累计和

    return events


def process_asset(args):
    """
    对单个资产组进行处理：
      1. 对该资产组的 'returns' 列（去掉缺失值）调用 symmetric_cusum_filter，得到事件列表；
      2. 筛选出事件中属于该资产的时间点；
      3. 在该资产的原始数据中提取这些事件时间对应的整行数据，并添加资产标签；
    参数:
      args: 一个元组 (asset, group, n)
        - asset: 资产名称；
        - group: DataFrame 对象，为 data.groupby('asset') 得到的当前资产的数据；
        - n: symmetric_cusum_filter 中的参数
    返回:
      对该资产采样后的 DataFrame
    """
    asset, group, n = args
    # 计算 'returns'（假设 data 中已经添加了 'log_close' 与 'returns' 列）
    returns = group['returns'].dropna()
    # 调用 cusum 过滤器得到事件列表（每个事件为 (timestamp, asset)）
    events = symmetric_cusum_filter(returns, n=n)
    # 筛选当前资产的事件时间点
    events_for_asset = [t for (t, a) in events if a == asset]
    # 从该资产的原始数据中提取对应行
    sampled_asset = group.loc[group.index.get_level_values('time').isin(events_for_asset)].copy()

    return sampled_asset


def generate_filter_df(data, sample_column='close', n=3, max_workers=None):
    """
    对 dataset 进行 cusum 采样，采用多进程并行处理各资产
    参数:
        data: 待采样的 DataFrame，要求包含 sample_column 列以及多级索引，其中一个级别为 'asset'
        sample_column: 要采样的列名（通常为 OHLC 中的某一列）
        n: symmetric_cusum_filter 的参数
        max_workers: 指定进程数（默认为 None，由系统决定）
    返回:
        filter_data: 经过采样并整理后，按资产为索引的 DataFrame
    """
    # 1. 计算 log 价格和收益率
    data['log_close'] = np.log(data[sample_column])
    data['returns'] = data.groupby('asset')['log_close'].diff()

    # 2. 构造任务列表，每个任务对应一个资产组
    tasks = [(asset, group, n) for asset, group in data.groupby('asset')]

    sampled_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for sampled_asset in executor.map(process_asset, tasks):
            sampled_list.append(sampled_asset)

    # 3. 合并所有资产的采样数据，并按索引排序
    sampled_data = pd.concat(sampled_list)
    sampled_data = sampled_data.sort_index()
    return sampled_data


if __name__ == '__main__':
    with open('15min_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data.columns)
    data = data.rename(columns=str.lower)
    data = data.rename_axis(index=['time', 'asset'])
    asset = ['ONEUSDT', 'TRXUSDT', 'BTCUSDT', 'ICXUSDT', 'HOTUSDT', 'BANDUSDT', 'FTMUSDT', 'CHZUSDT',
             'VETUSDT', 'XTZUSDT', 'ONTUSDT', 'WAVESUSDT', 'BCHUSDT', 'DUSKUSDT', 'ZECUSDT', 'NEOUSDT',
             'QTUMUSDT', 'DASHUSDT', 'BATUSDT', 'IOTXUSDT', 'ETHUSDT', 'ANKRUSDT', 'ZRXUSDT', 'RVNUSDT',
             'DENTUSDT', 'OMGUSDT', 'IOSTUSDT', 'ENJUSDT', 'DOGEUSDT', 'COSUSDT', 'FETUSDT', 'IOTAUSDT',
             'ADAUSDT', 'RENUSDT', 'ALGOUSDT', 'XMRUSDT', 'ETCUSDT', 'TROYUSDT', 'KAVAUSDT', 'LINKUSDT',
             'NULSUSDT', 'NKNUSDT', 'XRPUSDT', 'RLCUSDT', 'XLMUSDT', 'HBARUSDT', 'BNBUSDT', 'MTLUSDT',
             'ZILUSDT']
    data = data[[a in asset for a in data.index.get_level_values('asset')]]
    data = data[-len(data) // 2:-len(data) // 4]

    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    data['returns'] = u.returns(data)
    data['log_close'] = np.log(data['close'])
    # data['label'] = data.groupby(level='asset', group_keys=False)['close'].apply(triple_barrier_labeling)
    data['vwap'] = u.vwap(data)
    # 计算每个资产的对数收益率（当前与前一时点比较）
    # print(len(data))
    # data['alpha'] = f.alpha7(data)

    print(data)
    # 计算每个 asset 对应的列数（特征数量）
    # 计算每个 asset 在 MultiIndex DataFrame 中的行数
    # print(data['label'].value_counts())

    data = data.dropna()

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
    train_dict = {}
    for name, func in alpha_funcs:
        print(f"=== Now computing {name} in parallel... ===")
        data = compute_alpha_parallel(data, alpha_func=func, n_jobs=16)

    data = data.dropna()

    for col_name, func in alpha_funcs:
        print(data[col_name])
        daily_ic = data.groupby('asset').apply(lambda x: x[col_name].corr(x['future_return'], method='spearman'))
        print(daily_ic)
