import time

import numpy as np
import pandas as pd
from typing import List
from read_large_files import load_filtered_data_as_list, select_assets

from typing import List
import matplotlib.pyplot as plt


def calculate_S_value(x: np.ndarray) -> int:
    n = len(x)
    print(n)
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(x[j] - x[i])
    return S


def _mann_kendall_test(S: int, n: int) -> int:
    """
    对给定的序列 x 进行 Mann-Kendall 趋势检验。

    返回值:
        1 => 显著上升
       -1 => 显著下降
        0 => 无显著趋势
    """

    if n < 2:
        # 少于2条数据，不足以判定趋势
        return 0

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
    z_crit = 2.8
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
        S = [0] * len(closes)
        for i in range(len(closes)):
            # 取 i - window_size + 1 到 i 的数据作为子集
            start_idx = max(0, i - self.window_size + 1)
            if i < self.window_size:
                subset = closes[0: i + 1]
                S[i] = calculate_S_value(subset)
            else:
                prev_S = S[i - 1]
                for close in closes[start_idx: i]:
                    if closes[start_idx - 1] < close:
                        prev_S -= 1
                    else:
                        prev_S += 1
                    if closes[i] > close:
                        prev_S += 1
                    else:
                        prev_S -= 1
                S[i] = prev_S
            trend_signal = _mann_kendall_test(S[i], self.window_size)
            signals.append(trend_signal)
        df_asset["signal"] = signals
        return df_asset

    def generate_signal(self) -> List[pd.DataFrame]:
        ...
        # 1) 分组并计算信号
        grouped = self.df.groupby("asset", group_keys=False)
        df_list = []
        start = time.time()

        for asset, df_asset in grouped:
            df_result = self._compute_signals_for_asset(df_asset)
            df_list.append(df_result)
        end = time.time()
        print(end - start)
        print("start to concat")
        df_with_signal = pd.concat(df_list, ignore_index=True)

        # 2) 更新 self.df，让其带有最新的 signal
        #    这样 visualize_signals() 里的 self.df 才能用到
        self.df = df_with_signal

        # 3) 返回拆分后的列表（如果你还需要）
        new_dataset = []
        for original_df in self.original_dataset:
            original_df["time"] = pd.to_datetime(original_df["time"])
            merged_df = pd.merge(
                original_df,
                df_with_signal[["time", "asset", "signal"]],
                on=["time", "asset"],
                how="left"
            )
            new_dataset.append(merged_df)

        return new_dataset

    def visualize_signals(self, asset: str, start_date: str = None, end_date: str = None):
        """
        可视化指定资产在指定时间区间内的收盘价走势及 Mann-Kendall 信号。
        :param asset: 资产名称（如 "BTC-USDT"）
        :param start_date: 开始日期（字符串，格式 "YYYY-MM-DD"）
        :param end_date: 结束日期（字符串，格式 "YYYY-MM-DD"）
        """
        # 筛选出指定资产的数据
        df_asset = self.df[self.df["asset"] == asset].copy()
        if start_date:
            df_asset = df_asset[df_asset["time"] >= pd.to_datetime(start_date)]
        if end_date:
            df_asset = df_asset[df_asset["time"] <= pd.to_datetime(end_date)]

        if df_asset.empty:
            print(f"No data available for asset '{asset}' in the specified date range.")
            return

        # 绘制收盘价
        plt.figure(figsize=(14, 7))
        plt.plot(df_asset["time"], df_asset["close"], label="Close Price", color="lightblue", linewidth=2)

        # 绘制信号
        plt.scatter(
            df_asset["time"][df_asset["signal"] == 1],
            df_asset["close"][df_asset["signal"] == 1],
            label="Uptrend Signal",
            color="green",
            marker="^",
            alpha=1
        )
        plt.scatter(
            df_asset["time"][df_asset["signal"] == -1],
            df_asset["close"][df_asset["signal"] == -1],
            label="Downtrend Signal",
            color="red",
            marker="v",
            alpha=1
        )

        # 设置标题和标签
        plt.title(f"{asset} Price and Mann-Kendall Signals", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 显示图形
        plt.show()


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
    start_time = "2021-12-01"
    end_time = "2023-1-30"
    # asset_list = ['HOOK-USDT_future', 'ENS-USDT_future']
    asset_list = select_assets(future=True, n=50)
    filtered_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "15min")
    print("start initialize")
    strategy = MannKendallTrendByRow(filtered_data_list, window_size=50)
    print("start generate signal")
    strategy_result = strategy.generate_signal()
