import pickle

import pandas as pd

from read_large_files import load_filtered_data_as_list, select_assets
import time
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account, PositionManager, DefaultStopLossLogic, HoldNBarStopLossLogic, CostThresholdStrategy
from mann import MannKendallTrendByRow, filter_signals_by_daily_vectorized
from back_test_evaluation import PerformanceAnalyzer
from concurrent.futures import ProcessPoolExecutor


def process_asset_signals_wrapper(args):
    return process_asset_signals(*args)


def process_asset_signals(group, window, threshold, std_threshold):
    """
    处理单个资产的数据，生成信号列，并返回处理后的 DataFrame。
    """
    group = group.copy()  # 防止修改原始数据

    # 初始化 signal 列
    group["signal"] = 0
    # 计算基于百分比变化的收益率序列，用于波动率计算
    group["pct_change_high"] = group["close"].pct_change()

    for i in range(len(group)):
        if i < window:
            continue  # 窗口不足时跳过

        # 波动率筛选
        if std_threshold is not None:
            recent_period_length = int(window / 10)
            if i - recent_period_length >= 0:
                recent_slice = group.iloc[i - recent_period_length:i]
                recent_vol = recent_slice["pct_change_high"].std()
                # 如果近期波动率未超过阈值，则跳过此次循环
                if recent_vol > std_threshold:
                    continue

        current_close = group.iloc[i]["close"]
        past_window = group.iloc[i - window:i]
        past_max_high_idx = past_window["high"].idxmax()
        past_max_high = group.loc[past_max_high_idx, "close"]

        # 跳过最高价属于最近 window/10 根 K 线的情况
        recent_kline_limit = max(i - int(window / 10), i - window)
        if group.index.get_loc(past_max_high_idx) >= recent_kline_limit:
            continue

        # 检查是否满足条件：当前价格与过去最高价在指定阈值范围内
        condition_close_to_high = (current_close * (1 - threshold)
                                   <= past_max_high
                                   <= current_close * (1 + threshold))

        if condition_close_to_high:
            if group.iloc[i]["MA30"] < group.iloc[i]["MA5"]:
                continue
            group.iloc[i, group.columns.get_loc("signal")] = 1
            continue

        # --------------------------------------------
        past_window = group.iloc[i - window:i]
        past_min_low_idx = past_window["low"].idxmin()
        past_min_low = group.loc[past_min_low_idx, "close"]

        # 跳过最高价属于最近 window/10 根 K 线的情况
        recent_kline_limit = min(i - int(window / 10), i - window)
        if group.index.get_loc(past_min_low_idx) >= recent_kline_limit:
            continue

        # 检查是否满足条件：当前价格与过去最高价在指定阈值范围内
        condition_close_to_low = (current_close * (1 - threshold)
                                  <= past_min_low
                                  <= current_close * (1 + threshold))

        if condition_close_to_low:
            if group.iloc[i]["MA30"] > group.iloc[i]["MA5"]:
                continue
            group.iloc[i, group.columns.get_loc("signal")] = -1

    return group


def generate_signal(data, window, threshold, std_threshold=None):
    """
    使用多进程并行化生成交易信号，每个资产组独立一个进程计算 signal。
    """
    # 将数据按资产分组
    asset_groups = [group for _, group in data.groupby("asset")]

    # 为每个组准备参数
    params = [(group, window, threshold, std_threshold) for group in asset_groups]

    with ProcessPoolExecutor() as executor:
        # 使用顶层包装函数并行处理每个资产组
        results = executor.map(process_asset_signals_wrapper, params)

    # 合并所有结果
    result_df = pd.concat(results)
    # 可选：按索引排序以恢复原始顺序
    result_df = result_df.sort_index()

    return result_df


if __name__ == "__main__":
    start = "2024-2-1"
    end = "2024-7-30"

    assets = select_assets(spot=True, n=300)

    # assets = []
    data = load_filtered_data_as_list(start, end, assets, level="1d")

    strategy = DualMAStrategy(dataset=data, asset=assets, short=5, long=30)

    strategy.generate_signal()

    data = pd.concat(strategy.dataset, ignore_index=True)

    data = data.set_index(["time", "asset"])

    strategy_results = generate_signal(data.copy(), window=30, threshold=0.03, std_threshold=0.02)

    # #
    # strategy_results = pd.concat(strategy_results, ignore_index=True)
    # strategy_results = strategy_results.set_index(["time", "asset"])

    account = Account(initial_cash=100000)
    #stop = CostThresholdStrategy(gain_threshold=0.08, loss_threshold=0.08)
    stop = HoldNBarStopLossLogic(windows=7)
    broker = Broker(account, stop_loss_logic=stop)
    pos_manager = PositionManager(threshold=0.05)
    backtester = Backtest(broker, strategy_results, pos_manager)
    print("start_running")
    s = time.time()
    result = backtester.run()
    e = time.time()
    print(e - s)

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
