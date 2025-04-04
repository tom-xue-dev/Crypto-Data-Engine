import pickle
import sys
import numpy as np
import pandas as pd
from IC_calculator import compute_zscore, compute_ic
import alphalens as al
import utils as u
from Factor import *
from multiprocessing import Pool
from labeling import parallel_apply_triple_barrier
from Dataloader import DataLoader, DataLoaderConfig
from factor_evaluation import FactorEvaluator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def pca_transform(df, columns, n_components=2, prefix='pca'):
    """
    对指定列进行标准化 + PCA，并返回主成分 DataFrame。

    参数:
        df : pd.DataFrame
            原始数据
        columns : list
            要进行PCA的列名列表
        n_components : int
            要保留的主成分个数
        prefix : str
            生成的主成分列名前缀

    返回:
        pca_df : pd.DataFrame
            包含主成分的DataFrame，列名为 prefix_1, prefix_2, ...
    """
    # 取出要处理的列
    X = df[columns].copy()
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # PCA变换
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    # 构造结果DataFrame
    col_names = [f'{prefix}_{i + 1}' for i in range(n_components)]
    pca_values = pd.DataFrame(components, index=df.index, columns=col_names).to_numpy()
    return pca_values
import matplotlib.pyplot as plt


def factor_return_analysis_plot(df, factor_col, return_col, n_bins=5, title=None):
    """
    将因子列分为n组，输出并绘制每组的平均未来收益率

    参数：
        df         : 包含因子和收益率的 DataFrame
        factor_col : 因子列名
        return_col : 未来收益率列名
        n_bins     : 分组数量（如5为五分位）
        title      : 可选图标题

    返回：
        group_mean : 每组平均未来收益率（Pandas Series）
    """
    # 去掉缺失值
    df = df[[factor_col, return_col]].dropna()

    # 分组
    df['group'] = pd.qcut(df[factor_col], q=n_bins, labels=False, duplicates='drop')

    # 分组平均收益率
    group_mean = df.groupby('group')[return_col].mean()

    # 可视化
    plt.figure(figsize=(8, 5))
    group_mean.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title or f'{factor_col} future group return')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return group_mean


if __name__ == '__main__':
    config = DataLoaderConfig.load("load_config.yaml")
    data_loader = DataLoader(config)
    data = data_loader.load_all_data()
    data['returns'] = u.returns(data)
    data['log_close'] = np.log(data['close'])
    # data['label'] = parallel_apply_triple_barrier(data)
    data['future_return'] = data.groupby('asset')['close'].transform(lambda x: x.shift(-10) / x - 1)
    pd.set_option('display.max_columns', None)
    #data['label'] = np.where(data['future_return'] > 0, 1, 0)
    alpha_funcs = [
        ('alpha1', alpha1),
        ('alpha2', alpha2),
        ('alpha9', alpha9),
        ('alpha25', alpha25),
        ('alpha32', alpha32),
        ('alpha46', alpha46),
        ('alpha95', alpha95),
        ('alpha101', alpha101),
        # ('alpha102', alpha102),
        # ('alpha103', alpha103),
        # ('alpha104', alpha104),
        # ('alpha105', alpha105),
        # ('alpha106', alpha106),
        # ('alpha107', alpha107),
        # ('alpha108', alpha108),
        # ('alpha109',alpha109),
        # ('alpha112',alpha112),
        # ('alpha111',alpha111),
        # ('alpha112',alpha112),
        # ('alpha113',alpha113),
        # ('alpha114',alpha114),
        # ('alpha115',alpha115),
        # ('alpha116',alpha116),
        # ('alpha117',alpha117),

     ]
    for name, func in alpha_funcs:
        print(f"=== Now computing {name} in parallel... ===")
        data = compute_alpha_parallel(data, alpha_func=func, n_jobs=16)

    data = data.dropna()
    print(data)
    for col,_ in alpha_funcs:
        ic = compute_ic(data, feature_column=col,return_column='future_return')
        print(ic)
        factor_return_analysis_plot(data,col, 'future_return', n_bins=5, title=f'{col} future group return')