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
import numpy as np


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
    """
    对单个资产组的数据进行信号生成。
    """
    # 计算 Garman-Klass 波动率
    group = calculate_garman_klass_volatility(group, window)

    group["signal"] = 0  # 初始化信号列

    for i in range(len(group)):
        if i < window:
            continue  # 跳过窗口不足的前几天

        # 当前 close 和过去 window 天的最低价
        current_close = group.iloc[i]["high"]
        past_max_high_idx = group.iloc[i - window:i]["high"].idxmax()  # 找到最高价所在的索引
        past_max_high = group.loc[past_max_high_idx, "high"]  # 获取过去最高价

        # 跳过最高价属于最近 window / 10 根 K 线的情况
        recent_kline_limit = i - int(window / 10)
        if group.index.get_loc(past_max_high_idx) >= recent_kline_limit:
            continue  # 如果最高价的索引在最近 window/10 根 K 线内，跳过

        # 检查过去 window 根 K 线的 Garman-Klass 波动率是否低于阈值
        condition_volatility_low = None
        if volatility_threshold is not None:
            past_gk_volatility = group.iloc[i - int(window / 10):i]["GK_vol_rolling"].mean()
            condition_volatility_low = past_gk_volatility < volatility_threshold

        # 检查是否满足条件 1
        condition_close_to_low = current_close * (1 - threshold) <= past_max_high <= current_close * (
                1 + threshold
        )
        # if group.iloc[i]["MA200"] > group.iloc[i]["MA1200"]:
        #     continue
        # 如果所有条件满足，则标记信号
        if volatility_threshold is not None:
            if condition_close_to_low and condition_volatility_low:
                group.iloc[i, group.columns.get_loc("signal")] = 1
        else:
            if condition_close_to_low:
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
    start = "2023-1-1"
    end = "2024-11-30"

    assets = select_assets(spot=True, n=360)

    # assets = ["BTC-USDT_spot"]

    data = load_filtered_data_as_list(start, end, assets, level="15min")

    #strategy = DualMAStrategy(dataset=data, asset=assets, short=200, long=1200)

    #strategy.generate_signal()

    data = pd.concat(data, ignore_index=True)

    data = data.set_index(["time", "asset"])

    strategy_results = generate_signal(data.copy(), window=1200, threshold=0.005, volatility_threshold=1e-05)

    # #
    # strategy_results = pd.concat(strategy_results, ignore_index=True)
    # strategy_results = strategy_results.set_index(["time", "asset"])

    account = Account(initial_cash=100000)
    stop = CostThresholdStrategy(gain_threshold=0.08, loss_threshold=0.08)
    #stop = HoldNBarStopLossLogic(windows=105)
    #stop = DefaultStopLossLogic(max_drawdown=0.08)
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
