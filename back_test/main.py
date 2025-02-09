import pickle
import sys

import pandas as pd

from read_large_files import load_filtered_data_as_list, select_assets
import time
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account, PositionManager, DefaultStopLossLogic, HoldNBarStopLossLogic, CostThresholdStrategy, \
    CostATRStrategy,AtrPositionManager
from mann import MannKendallTrendByRow, filter_signals_by_daily_vectorized
from back_test_evaluation import PerformanceAnalyzer
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def build_segments_n(dataset: pd.DataFrame, n: int) -> list[tuple[float, float]]:
    """
    将 dataset 按 time 维度切分成 2^n 个小块，
    每块计算 (min_low, max_high) 并返回列表。
    假设:
      - MultiIndex: (time, asset)
      - 列包含 ['low', 'high']
      - len(unique_times) >= 2^n
    """
    times = dataset.index.get_level_values('time').unique().sort_values()
    total_times = len(times)
    num_segments = 2 ** n  # 要切分的段数

    if total_times < num_segments:
        raise ValueError(f"time 数量({total_times})不足以切分成 {num_segments} 份。")

    step = total_times // num_segments  # 每段大小 (可能有余数)

    segments = []
    start = 0
    for i in range(num_segments):
        # 最后一段取到结尾，处理可能的余数
        if i == num_segments - 1:
            end = total_times
        else:
            end = start + step

        # 切分 times
        seg_times = times[start:end]
        sub_df = dataset.loc[seg_times]

        min_low = sub_df['low'].min()
        max_high = sub_df['high'].max()
        segments.append((min_low, max_high))

        start = end

    return segments


def compare_and_merge(segA: tuple[float, float], segB: tuple[float, float]) -> tuple[
    tuple[bool, bool], tuple[float, float]]:
    """
    比较 segA 与 segB 的 (min, max)，返回:
    1) (is_min_lower, is_max_lower)
    2) 合并后的新 segment (merged_min, merged_max)
    """
    minA, maxA = segA
    minB, maxB = segB
    is_min_lower = (minA < minB)
    is_max_lower = (maxA < maxB)

    # 合并后新的 segment
    merged_min = min(minA, minB)
    merged_max = max(maxA, maxB)
    return (is_min_lower, is_max_lower), (merged_min, merged_max)


def compare_segments_n(segments: list[tuple[float, float]]) -> list[tuple[bool, bool]]:
    """
    给定最底层 (2^n 个) segments，使用自底向上的两两合并，
    每合并一次就记录一个 (is_min_lower, is_max_lower)。
    最终返回所有比较结果的列表，总长度为 2^n - 1。
    """
    results = []
    layer = segments[:]

    # 不断两两合并，直到只剩一个
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            segA = layer[i]
            segB = layer[i + 1]
            (is_min_lower, is_max_lower), merged_seg = compare_and_merge(segA, segB)
            results.append((is_min_lower, is_max_lower))
            next_layer.append(merged_seg)
        layer = next_layer
    return results


def check_halves_n(dataset: pd.DataFrame, n: int = 4) -> list[tuple[bool, bool]]:
    """
    综合入口:
    1) 先按 time 切成 2^n 份，得到每份 (min_low, max_high)
    2) 再自底向上合并并比较，返回 (is_min_lower, is_max_lower) 的列表
       共会返回 2^n - 1 次比较结果
    """
    # 1) 建立 2^n 份 segments
    segments = build_segments_n(dataset, n=n)
    # 2) 自底向上合并并比较
    results = compare_segments_n(segments)
    return results


def process_asset_signals_wrapper(args):
    return process_asset_signals(*args)


def calculate_garman_klass_volatility(group, window):
    """
    在 DataFrame 中添加 Garman-Klass 波动率列。
    """
    group['GK_vol'] = (
            0.5 * (np.log(group['high'] / group['low'])) ** 2 -
            (2 * np.log(2) - 1) / window * (np.log(group['close'] / group['open'])) ** 2
    )
    group['GK_vol_rolling'] = group['GK_vol'].rolling(window=window).mean()
    return group


def process_asset_signals(group, window, threshold, volatility_threshold):
    group["signal"] = 0  # 初始化信号列

    for i in range(len(group)):
        if i - int(window / 50) < window * 5:
            continue  # 跳过窗口不足的前几天

        recent_kline_limit = i - int(window / 100)

        condition_long = False
        sub_df = group.iloc[i - window:i]
        current_close_long = group.iloc[i]["close"]
        ans = check_halves_n(sub_df)
        count_first_true = sum(1 for x in ans if x[0])
        count_sec_true = sum(1 for x in ans if x[1])
        # print(f"第1个元素为 True 的元组数量: {count_first_true}")
        # print(f"第2个元素为 True 的元组数量: {count_sec_true}")
        if count_first_true > 10 and count_sec_true > 10:
            condition_long = True
        over_look_past_long = group.iloc[i - window * 5:i]["close"].idxmax()
        # 检查是否满足条件 1
        # 当前收盘必须大于等于之前的最低收盘价
        current_close_short = group.iloc[i]["low"]
        past_min_low_idx = group.iloc[i - window:i]["low"].idxmin()  # 找到最高价所在的索引
        past_min_low = group.loc[past_min_low_idx, "low"]  # 获取过去最高价

        condition_short = past_min_low >= current_close_short * (
                1 + threshold
        )

        if condition_long and condition_short:
            continue

        if condition_long:
            # if group.index.get_loc(past_max_high_idx) >= recent_kline_limit:
            #     continue  # 如果最高价的索引在最近 window/10 根 K 线内，跳过
            if group.iloc[i]["MA50"] < group.iloc[i - 50]["MA50"]:
                continue
            if group.loc[over_look_past_long, "high"] * 0.9 < current_close_long < group.loc[
                over_look_past_long, "high"]:  # 超远搜索过去2个月的最高价格
                continue
            past_10_signals = group.iloc[max(0, i - 10):i]["signal"]
            if 1 in past_10_signals.values:
                continue
            group.iloc[i, group.columns.get_loc("signal")] = 1

    return group


def generate_signal(data, window, threshold, volatility_threshold=None):
    """
    使用多进程并行化生成交易信号，每个资产组独立一个进程计算 signal。
    """
    # 将数据按资产分组
    asset_groups = [group for _, group in data.groupby("asset")]

    # 为每个组准备参数
    params = [(group, window, threshold, volatility_threshold) for group in asset_groups]

    with ProcessPoolExecutor() as executor:
        # 使用顶层包装函数并行处理每个资产组
        results = executor.map(process_asset_signals_wrapper, params)

    # 合并所有结果
    result_df = pd.concat(results)
    # 可选：按索引排序以恢复原始顺序
    result_df = result_df.sort_index()

    return result_df


if __name__ == "__main__":
    # start = "2024-1-1"
    # end = "2024-11-30"
    #
    # assets = select_assets(spot=True, n=160)
    #
    # #assets = []
    #
    # data = load_filtered_data_as_list(start, end, assets, level="15min")
    # strategy = DualMAStrategy(dataset=data, asset=assets, short=50, long=3)
    #
    #
    # data = pd.concat(strategy.dataset, ignore_index=True)
    #
    # data = data.set_index(["time", "asset"])
    #
    # strategy_results = generate_signal(data, window=1200, threshold=0.01)

    # #
    # strategy_results = pd.concat(strategy_results, ignore_index=True)
    # strategy_results = strategy_results.set_index(["time", "asset"])
    with open("data.pkl", "rb") as file:
        strategy_results = pickle.load(file)
    #strategy_results = strategy_results[:len(strategy_results)//2]
    print(len(strategy_results['feature']),len(strategy_results['label']))
    sys.exit(0)
    # strategy_results = strategy_results.set_index(["time", "asset"])
    account = Account(initial_cash=13000)
    #stop = CostThresholdStrategy(gain_threshold=0.2, loss_threshold=0.2)
    stop = HoldNBarStopLossLogic(windows=100)
    #stop = DefaultStopLossLogic(max_drawdown=0.05)
    #stop = CostATRStrategy()
    broker = Broker(account, stop_loss_logic=stop)
    #pos_manager = AtrPositionManager(risk_percent=0.05,loss_times=1)
    pos_manager = PositionManager(threshold=0.1)
    backtester = Backtest(broker, strategy_results, pos_manager)
    print("start_running")
    s = time.time()
    result = backtester.run()
    e = time.time()

    analyser = PerformanceAnalyzer(result["net_value_history"])

    analyser.plot_net_value()

    print(analyser.summary())

    with open("transactions.txt", "w") as f:
        for transaction in backtester.broker.account.transactions:
            f.write(str(transaction) + "\n")

    with open("result.txt", "w") as f:
        f.write(
            result["net_value_history"].to_string(
                formatters={
                    'net_value': '{:.2f}'.format,  # net_value 列保留两位小数
                    'cash': '{:.2f}'.format  # cash 列保留两位小数
                },
                index=False  # 可选，是否打印索引
            )
        )
