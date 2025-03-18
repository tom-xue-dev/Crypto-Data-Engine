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
    with open("data_signal.pkl", "rb") as file:
        strategy_results = pickle.load(file)
    #strategy_results = strategy_results[:len(strategy_results)//2]
    # strategy_results = strategy_results.set_index(["time", "asset"])
    # for asset,group in strategy_results.groupby('asset'):
    #     print(asset)
    account = Account(initial_cash=13000)
    #stop = CostThresholdStrategy(gain_threshold=0.03, loss_threshold=0.03)
    stop = HoldNBarStopLossLogic(windows=10000)
    #stop = DefaultStopLossLogic(max_drawdown=0.05)
    #stop = CostATRStrategy()
    broker = Broker(account, stop_loss_logic=stop)
    #pos_manager = AtrPositionManager(risk_percent=0.05,loss_times=1)
    pos_manager = PositionManager(threshold=0.1)
    backtester = Backtest(broker, strategy_results, pos_manager)
    # print("start_running")
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
