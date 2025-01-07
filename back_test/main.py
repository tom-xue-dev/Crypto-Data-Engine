import pandas as pd

from read_large_files import load_filtered_data_as_list, select_assets
import time
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account, PositionManager, DefaultStopLossLogic,HoldNBarStopLossLogic,CostThresholdStrategy
from mann import MannKendallTrendByRow, filter_signals_by_daily_vectorized
from back_test_evaluation import PerformanceAnalyzer

start_time = "2023-12-01"
end_time = "2024-1-30"

asset_list = ["BTC-USDT_spot"]
min_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "15min")

strategy = DualMAStrategy(dataset=min_data_list, asset=asset_list, long=10, short=5)
strategy.generate_signal()

strategy_results = strategy.get_dataset()
#
strategy_results = pd.concat(strategy_results, ignore_index=True)
strategy_results = strategy_results.set_index(["time", "asset"])

account = Account(initial_cash=10000)
stop = CostThresholdStrategy(gain_threshold=0.15,loss_threshold=0.02)
broker = Broker(account, stop_loss_logic=stop)
pos_manager = PositionManager(threshold=1)
backtester = Backtest(broker, strategy_results, pos_manager)
print("start_running")
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
