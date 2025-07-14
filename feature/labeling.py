import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def apply_triple_barrier(group):
    # group 为 DataFrame，该组对应单个资产
    # 这里我们只取该组的 'close' 列传入 triple_barrier_labeling
    return triple_barrier_labeling(group['close'])


def parallel_apply_triple_barrier(df):
    """
    按照 index 中的 'asset' 分组，对每个组并行计算 triple_barrier_labeling，
    返回合并后的 Series，该 Series 的 index 与原 DataFrame 对齐。
    """
    # 按照 'asset' 分组；注意这里 df.index 是 MultiIndex，假设 'asset' 是其中一个层级
    groups = [group for _, group in df.groupby(level='asset')]

    with ProcessPoolExecutor() as executor:
        # executor.map 并行地将每个组传入 apply_triple_barrier 函数
        results = list(executor.map(apply_triple_barrier, groups))

    # 将所有组的结果合并，并根据索引排序对齐原 DataFrame
    return pd.concat(results).sort_index()


def triple_barrier_labeling(prices, upper_pct=0.03, lower_pct=0.03, max_time=50):
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
            labels[t] = 2  # 触碰下界
        else:
            labels[t] = 0  # 仅时间屏障触发

    return pd.Series(labels, index=price_idx)
