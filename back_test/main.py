from read_large_files import load_filtered_data_as_list
import time
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account

start_time = "2020-12-01"
end_time = "2021-6-30"
asset_list = ['BNB-USDT_spot', 'BTC-USDT_spot', 'ETH-USDT_spot', 'NEO-USDT_spot']  # 替换为您需要的资产

filtered_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "15min")

strategy = DualMAStrategy(filtered_data_list, asset_list, 10, 5)
strategy.generate_signal()
strategy_results = strategy.get_dataset()

account = Account(initial_cash=10000)

broker = Broker(account)

backtester = Backtest(broker, strategy_results)

start = time.time()
result = backtester.run()
end = time.time()

print(end-start)

print(result["net_value_history"])
