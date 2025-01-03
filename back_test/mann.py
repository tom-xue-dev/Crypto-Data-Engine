import time

import numpy as np
import pandas as pd
from typing import List
from read_large_files import load_filtered_data_as_list

from typing import List


def _mann_kendall_test(x: np.ndarray) -> int:
    """
    对给定的序列 x 进行 Mann-Kendall 趋势检验。

    返回值:
        1 => 显著上升
       -1 => 显著下降
        0 => 无显著趋势
    """
    n = len(x)
    if n < 2:
        # 少于2条数据，不足以判定趋势
        return 0

    # 1) 计算统计量 S
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(x[j] - x[i])

    # 2) 计算方差 Var(S) - 无重复值的简单公式
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # 3) 计算 Z (带连续性修正)
    if S > 0:
        z = (S - 1) / np.sqrt(var_s)
    elif S == 0:
        z = 0
    else:
        z = (S + 1) / np.sqrt(var_s)

    # 4) 临界值（双侧检验, alpha=0.05 => z_crit=1.96）
    z_crit = 1.96
    if abs(z) > z_crit:
        return 1 if z > 0 else -1
    else:
        return 0


class MannKendallTrendByRow:
    def __init__(self,
                 dataset: List[pd.DataFrame],
                 window_size: int = 7):
        """
        dataset: 外层是 List，每个元素是一个 DataFrame，
                 每个 DataFrame 的列包括 [time, asset, open, high, low, close]。
        window_size: Mann-Kendall 检验所用的滚动窗口大小(n)。
                     对于第 i 条数据，只看 i - n + 1 到 i 的数据来判定趋势。
        """
        if not isinstance(dataset, list):
            raise ValueError("dataset 必须是一个列表(List)，其中每个元素都是一个 DataFrame。")
        if not all(isinstance(df, pd.DataFrame) for df in dataset):
            raise ValueError("dataset 列表中每个元素必须都是 pandas.DataFrame。")

        self.window_size = window_size  # 加入到类属性，方便后续使用

        # 1) 把所有 DataFrame 合并成一个大的 DataFrame
        self.original_dataset = dataset  # 保存原始引用，以便后面要回填signal
        self.df = pd.concat(dataset, ignore_index=True)

        # 检查必要字段
        required_cols = {"time", "asset", "close"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"数据缺少必要列: {required_cols}")

        # 统一转换 time 为 datetime，并按照 (asset, time) 排序
        self.df["time"] = pd.to_datetime(self.df["time"])
        self.df.sort_values(["asset", "time"], ascending=[True, True], inplace=True)

        # 准备一列来存放信号
        self.df["signal"] = 0

    def _compute_signals_for_asset(self, df_asset: pd.DataFrame) -> pd.DataFrame:
        """
        针对同一资产的所有行，按照时间先后，逐行截取 window_size 个数据(或更少)，
        做 Mann-Kendall，得到信号序列。
        返回: 带有 signal 列的 df_asset (保持原顺序和索引)。
        """
        closes = df_asset["close"].values
        signals = []
        for i in range(len(closes)):
            # 取 i - window_size + 1 到 i 的数据作为子集
            start_idx = max(0, i - self.window_size + 1)
            subset = closes[start_idx: i + 1]
            trend_signal = _mann_kendall_test(subset)
            signals.append(trend_signal)

        df_asset["signal"] = signals
        return df_asset

    def generate_signal(self) -> List[pd.DataFrame]:
        """
        逐行计算 Mann-Kendall 趋势信号，并将结果回填到每个DataFrame中（在其末尾增加一列 'signal'）。
        返回: 一个新的 List[pd.DataFrame]，与输入 dataset 对应的形状，且多一列 'signal'。
        """
        # 1) 分组，对每个 asset 分别做 逐行滚动窗口 的 Mann-Kendall

        grouped = self.df.groupby("asset", group_keys=False)

        df_list = []
        for asset, df_asset in grouped:
            # 这里直接在循环里调用你的处理函数
            df_result = self._compute_signals_for_asset(df_asset)
            df_list.append(df_result)

        df_with_signal = pd.concat(df_list, ignore_index=True)

        # 2) 按原始 DataFrame 拆分并合并回 'signal'
        new_dataset = []
        for original_df in self.original_dataset:
            # 确保 original_df 里的 time 列也是 datetime 类型
            original_df["time"] = pd.to_datetime(original_df["time"])

            merged_df = pd.merge(
                original_df,
                df_with_signal[["time", "asset", "signal"]],
                on=["time", "asset"],
                how="left"
            )
            new_dataset.append(merged_df)

        return new_dataset


def analyze_future_returns_all_signals(df_list, n=3):
    """
    df_list: 包含多个 DataFrame 的列表，每个 DataFrame 至少包含
             ['time', 'open', 'high', 'low', 'close', 'asset', 'signal']
    n:       向后看的 K 线条数 (第 n 条)
    """

    # 1. 合并所有 df
    all_df = pd.concat(df_list, ignore_index=True)

    # 2. 对 (asset) 分组 & 按 time 排序
    grouped = all_df.groupby('asset', group_keys=False)

    # 用于收集所有 signal=1 对应的“第 n 条 k 线的涨跌幅”
    all_n_returns = []

    # 3. 遍历每一个资产的小表
    for asset, group in grouped:
        group = group.sort_values('time').reset_index(drop=True)

        # 找到 signal=1 的行
        signal_rows = group[group['signal'] == 1]
        if signal_rows.empty:
            continue

        # 逐行处理 signal=1 的索引
        for idx in signal_rows.index:
            current_close = group.loc[idx, 'close']

            # 获取第 n 条 k 线所在的行索引
            target_idx = idx + n
            if target_idx >= len(group):
                # 不足 n 条则跳过
                continue

            nth_close = group.loc[target_idx, 'close']
            nth_return = (nth_close - current_close) / current_close
            all_n_returns.append(nth_return)

    # 4. 汇总结果
    if len(all_n_returns) == 0:
        return {
            'count': 0,
            'up_probability': None,
            'average_return': None
        }
    else:
        # 上涨概率
        up_probability = sum(r > 0 for r in all_n_returns) / len(all_n_returns)
        # 平均涨跌幅
        average_return = sum(all_n_returns) / len(all_n_returns)

        return {
            'count': len(all_n_returns),
            'up_probability': up_probability,
            'average_return': average_return
        }


# ------------------ 使用示例 ------------------ #
if __name__ == "__main__":
    # 模拟 dataset
    start_time = "2017-12-01"
    end_time = "2024-6-30"
    asset_list = ['DOGE-USDT_future']  # 替换为您需要的资产

    filtered_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "1d")
    print("start initialize")

    mk_detector = MannKendallTrendByRow(filtered_data_list, window_size=28)
    print("start generate signal")

    new_dataset_with_signal = mk_detector.generate_signal()

    for i in range(2, 60):
        result = analyze_future_returns_all_signals(new_dataset_with_signal, n=i)
        print(i, result)
