from read_large_files import load_filtered_data_as_list, select_assets
import time
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account, PositionManager, DefaultStopLossLogic
from mann import MannKendallTrendByRow
from back_test_evaluation import PerformanceAnalyzer

start_time = "2023-12-01"
end_time = "2024-6-30"
# asset_list = select_assets(future=True, n=10)  # 替换为您需要的资产
asset_list = ['BTC-USDT_spot','ETH-USDT_spot']
filtered_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "15min")

mk_detector = MannKendallTrendByRow(filtered_data_list, window_size=14)
strategy_results = mk_detector.generate_signal()

print("strategy over")
account = Account(initial_cash=10000)
stop = DefaultStopLossLogic(max_drawdown=0.08)
broker = Broker(account, stop_loss_logic=stop)
pos_manager = PositionManager(threshold=0.3)
backtester = Backtest(broker, strategy_results, pos_manager)

result = backtester.run()

analyser = PerformanceAnalyzer(result["net_value_history"])

analyser.plot_net_value()

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
