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
    with open("test.pkl", "rb") as file:
        strategy_results = pickle.load(file)

    for asset,group in strategy_results.groupby('asset'):
        account = Account(initial_cash=10000)
        #stop = CostThresholdStrategy(gain_threshold=0.1, loss_threshold=0.05,windows=20)
        stop = HoldNBarStopLossLogic(windows=20)
        #stop = DefaultStopLossLogic(max_drawdown=0.1)
        #stop = CostATRStrategy()
        broker = Broker(account, stop_loss_logic=stop,fees=0.001)
        # pos_manager = AtrPositionManager(risk_percent=0.05,loss_times=1)
        pos_manager = PositionManager(threshold=1)
        backtester = Backtest(broker, group, pos_manager)
        # print("start_running")
        s = time.time()
        result = backtester.run()
        e = time.time()
        print(e-s)
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
