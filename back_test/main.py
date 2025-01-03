from read_large_files import load_filtered_data_as_list
import time
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account, PositionManager
from mann import MannKendallTrendByRow

start_time = "2017-12-01"
end_time = "2024-6-30"
asset_list = ['BTC-USDT_spot']  # 替换为您需要的资产

filtered_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "1d")


mk_detector = MannKendallTrendByRow(filtered_data_list, window_size=7)
strategy_results = mk_detector.generate_signal()

print("strategy over")
account = Account(initial_cash=10000)

broker = Broker(account)
pos_manager = PositionManager(threshold=0.15)
backtester = Backtest(broker, strategy_results, pos_manager)

result = backtester.run()

print(result["net_value_history"])
with open("transactions.txt", "w") as f:
    for transaction in backtester.broker.account.transactions:
        f.write(str(transaction) + "\n")

with open("result.txt", "w") as f:
    f.write(str(result["net_value_history"].to_string()))
