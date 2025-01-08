import pickle

import pandas as pd

from read_large_files import load_filtered_data_as_list, select_assets
import time
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account, PositionManager, DefaultStopLossLogic, HoldNBarStopLossLogic, CostThresholdStrategy
from mann import MannKendallTrendByRow, filter_signals_by_daily_vectorized
from back_test_evaluation import PerformanceAnalyzer


def generate_signal(data, window, threshold, std_threshold=None):
    """
    生成交易信号列，基于过去 window 天内的最高价和当前 close 价格的比较。

    参数:
        data: DataFrame, 数据集
        window: int, 回看窗口天数
        threshold: float, 阈值（百分比，如 0.1 表示 10%）

    返回:
        带有 signal 列的 DataFrame
    """
    # 添加 signal 列，初始化为 0
    data["signal"] = 0

    # 对每个资产进行操作
    for asset, group in data.groupby("asset"):
        group = group.copy()  # 防止修改原始数据

        for i in range(len(group)):
            if i < window:
                continue  # 跳过窗口不足的前几天

            # 当前 close 和过去 window 天的最高价
            current_high = group.iloc[i]["high"]
            past_max_high_idx = group.iloc[i - window:i]["high"].idxmax()  # 找到最高价所在的索引
            past_max_high = group.loc[past_max_high_idx, "high"]  # 获取过去最高价

            current_close = group.iloc[i]["low"]
            past_min_low_idx = group.iloc[i - window:i]["low"].idxmin()  # 找到最低价所在的索引
            past_min_low = group.loc[past_min_low_idx, "low"]
            # 跳过最高价属于最近 window / 10 根 K 线的情况


            # 检查是否满足条件 1
            condition_long = current_high * (1 - threshold) <= past_max_high <= current_high * (
                    1 + threshold
            )

            condition_short = current_close * (1 - threshold) <= past_min_low <= current_close * (
                    1 + threshold
            )
            # 如果所有条件满足，则标记信号
            if std_threshold is not None:
                pass
            else:
                if condition_long:
                    recent_kline_limit = max(i - int(window / 10), i - window)
                    if group.index.get_loc(past_max_high_idx) >= recent_kline_limit:
                        continue  # 如果最高价的索引在最近 window/10 根 K 线内，跳过
                    group.iloc[i, group.columns.get_loc("signal")] = 1
                if  condition_short:
                    recent_kline_limit = min(i - int(window / 10), i - window)
                    if group.index.get_loc(past_min_low_idx) <= recent_kline_limit:
                        continue
                    group.iloc[i, group.columns.get_loc("signal")] = -1

        data.loc[group.index, "signal"] = group["signal"]

    return data

start = "2024-1-1"
end = "2024-11-30"

assets = select_assets(spot=True, n=300)

# assets = []
data = load_filtered_data_as_list(start, end, assets, level="1d")

data = pd.concat(data, ignore_index=True)

data = data.set_index(["time", "asset"])

strategy_results = generate_signal(data.copy(), window=30, threshold=0.01)

# #
# strategy_results = pd.concat(strategy_results, ignore_index=True)
# strategy_results = strategy_results.set_index(["time", "asset"])

account = Account(initial_cash=100000)
stop = CostThresholdStrategy(gain_threshold=0.08, loss_threshold=0.08)
# stop = HoldNBarStopLossLogic(windows=1)
broker = Broker(account, stop_loss_logic=stop)
pos_manager = PositionManager(threshold=0.02)
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
