import pickle
import sys
import numpy as np
from read_large_files import map_and_load_pkl_files, select_assets
import pandas as pd
from IC_calculator import compute_zscore, compute_ic
import alphalens as al
import utils as u
from Factor import alpha1, alpha2, alpha25, alpha32, alpha46, alpha51, alpha95, alpha9, alpha103, alpha35, alpha101, \
    alpha104, alpha106, alpha105, alpha102, alpha107, alpha108
from multiprocessing import Pool
from labeling import parallel_apply_triple_barrier


def downside_volatility_ratio(df, period):
    df['squared_returns'] = df['returns'] ** 2  # 计算收益平方
    df['downside_squared'] = df['squared_returns'] * (df['returns'] < 0)  # 仅保留下行波动的平方部分
    # 计算滚动窗口内的下行波动占比
    df['downside_ratio'] = df['downside_squared'].rolling(period).sum() / df['squared_returns'].rolling(period).sum()
    return df


def compute_alpha_parallel(data, alpha_func, n_jobs=4):
    """
    多进程: 对 data 按 asset 分组, 并行执行 alpha_func(df), 最终合并结果
    :param data: 包含至少 ['asset'] 的 DataFrame
    :param alpha_func: 一个函数, 入参为 df(单个asset的数据), 出参仍是 df
    :param n_jobs: 进程数
    :return: 添加了 alphaXX 列的 DataFrame
    """
    # 按 asset 分组，拆分DataFrame
    grouped = []
    for asset, df_asset in data.groupby('asset'):
        grouped.append(df_asset)

    # 多进程映射
    with Pool(n_jobs) as pool:
        results = pool.map(alpha_func, grouped)

    # 合并
    df_out = pd.concat(results, axis=0)
    df_out.sort_index(inplace=True)
    return df_out


if __name__ == '__main__':
    with open('15min_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data.columns)
    data = data.rename(columns=str.lower)
    data = data.rename_axis(index=['time', 'asset'])
    start = "2020-1-1"
    end = "2022-12-31"
    # assets = select_assets(start_time=start, spot=True, n=50)
    # data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")
    # print(data.columns)

    asset = ['ONEUSDT', 'TRXUSDT', 'BTCUSDT', 'ICXUSDT', 'HOTUSDT', 'BANDUSDT', 'FTMUSDT', 'CHZUSDT',
             'VETUSDT', 'XTZUSDT', 'ONTUSDT', 'WAVESUSDT', 'BCHUSDT', 'DUSKUSDT', 'ZECUSDT', 'NEOUSDT',
             'QTUMUSDT', 'DASHUSDT', 'BATUSDT', 'IOTXUSDT', 'ETHUSDT', 'ANKRUSDT', 'ZRXUSDT', 'RVNUSDT',
             'DENTUSDT', 'OMGUSDT', 'IOSTUSDT', 'ENJUSDT', 'DOGEUSDT', 'COSUSDT', 'FETUSDT', 'IOTAUSDT',
             'ADAUSDT', 'RENUSDT', 'ALGOUSDT', 'XMRUSDT', 'ETCUSDT', 'TROYUSDT', 'KAVAUSDT', 'LINKUSDT',
             'NULSUSDT', 'NKNUSDT', 'XRPUSDT', 'RLCUSDT', 'XLMUSDT', 'HBARUSDT', 'BNBUSDT', 'MTLUSDT',
             'ZILUSDT']

    #asset = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'TRXUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT']
    data = data[[a in asset for a in data.index.get_level_values('asset')]]
    data = data[:len(data) // 8]
    # with open("origin_data.pkl","wb") as f:
    #     pickle.dump(data,f)
    data['returns'] = u.returns(data)
    data['log_close'] = np.log(data['close'])
    print(data)
    # data['label'] = parallel_apply_triple_barrier(data)
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    #data['label'] = np.where(data['future_return'] > 0, 1, 0)
    alpha_funcs = [
        # ('alpha1', alpha1),
        # ('alpha2', alpha2),
        # ('alpha9', alpha9),
        # ('alpha25', alpha25),
        # ('alpha32', alpha32),
        # ('alpha46', alpha46),
        # ('alpha95', alpha95),
        # ('alpha101', alpha101),
        # ('alpha102', alpha102),
        # ('alpha103', alpha103),
        # ('alpha104', alpha104),
        # ('alpha105', alpha105),
        # ('alpha106', alpha106),
        # ('alpha107', alpha107),
        # ('alpha108', alpha108)
    ]

    for name, func in alpha_funcs:
        print(f"=== Now computing {name} in parallel... ===")
        data = compute_alpha_parallel(data, alpha_func=func, n_jobs=16)

