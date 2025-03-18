import pickle
import sys
import numpy as np
from statsmodels.tsa.stattools import adfuller

from read_large_files import map_and_load_pkl_files, select_assets
import pandas as pd
from IC_calculator import compute_zscore, compute_ic
from CUSUM_filter import generate_filter_df
import utils as u
from Factor import alpha1, alpha2, alpha25, alpha32, alpha46, alpha51, alpha95, alpha9, alpha103, alpha35, alpha101, \
    alpha104, alpha106, alpha105, alpha102, alpha107, alpha108
from multiprocessing import Pool
from labeling import parallel_apply_triple_barrier
import statsmodels.api as sm
from support import heikin_ashi_transform,process_wrapper,ma_smooth


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


def autocorrelation(x):
    n = len(x)
    x = np.array(x)
    result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    acf = result[result.size // 2:] / (np.var(x) * np.arange(n, 0, -1))
    return acf


if __name__ == '__main__':
    start = "2020-1-1"
    end = "2022-12-31"
    assets = select_assets(start_time=start, spot=True, n=20)
    print(assets)

    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")

    print(data)
