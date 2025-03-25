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
from read_data import DataLoader
from factor_evaluation import FactorEvaluator
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
    asset_list = ['DCRUSDT']
    data_loader = DataLoader(asset_list = asset_list,file_end='USDT')
    data = data_loader.load_all_data()
    data['returns'] = u.returns(data)
    data['log_close'] = np.log(data['close'])
    print(data)
    # data['label'] = parallel_apply_triple_barrier(data)
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    #data['label'] = np.where(data['future_return'] > 0, 1, 0)
    alpha_funcs = [
        ('alpha1', alpha1),
        ('alpha2', alpha2),
        #('alpha9', alpha9),
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
    pd.set_option('display.max_columns', None)
    print(data)
    df = data.dropna()
    ic = df[['alpha1', 'future_return']].dropna().corr(method='pearson').iloc[0, 1]
    print(ic)
    # 2. 分层
    # evaluator = FactorEvaluator(data)
    # evaluator.plot_factor_distribution(factor_column='returns', upper_bound=10, lower_bound=0)
    # evaluator.plot_cumulative_ic(factor_column='alpha1', return_column='future_return',method='pearson')
    # evaluator.plot_cumulative_ic(factor_column='alpha2', return_column='future_return', method='pearson')
