# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import math
#
# from Account import Account
# from strategy import MovingAverageStrategy
# from backtest_simulation import Backtest
#
# # 生成测试数据
# # 假设我们有2天数据，每天3个时间点（如9:00,10:00,11:00），仅有一种资产 AAPL
# dates = [datetime(2024,1,1,9), datetime(2024,1,1,10), datetime(2024,1,1,11),
#          datetime(2024,1,2,9)]
#
# prices = [150, 100, 300, 120]  # 模拟价格小幅波动
# asset = ['AAPL'] * len(dates)
#
# data = pd.DataFrame({
#     'time': dates,
#     'asset': asset,
#     'open': prices,
#     'high': [p+1 for p in prices],
#     'low': [p-1 for p in prices],
#     'close': prices
# })
#
# # 将数据拆分为每日DataFrame列表（符合strategy使用格式）
# day1 = data[data['time'].dt.date == datetime(2024,1,1).date()].copy()
# day2 = data[data['time'].dt.date == datetime(2024,1,2).date()].copy()
# dataset = [day1, day2]
# print(dataset)
# # 创建策略实例，period=3
# strategy = MovingAverageStrategy(dataset, ['AAPL'], 3)
# strategy.calculate_MA(2)
# signal_data = strategy.generate_signal()
#
# print("策略生成信号后的数据：")
# for df in signal_data:
#     print(df)
#
# # 假设回测使用的参数
# initial_capital = 10000
# asset_parameters = {
#     # 'AAPL': {
#     #     'leverage': 1,
#     #     'hourly_rate': 0.001,
#     #     'min_trade_unit': 0.01
#     # }
# }
# stop_loss_threshold = 0.05
#
# # 创建backtest实例并运行
# backtester = Backtest(
#     strategy_results=signal_data,
#     initial_capital=initial_capital,
#     asset_parameters=asset_parameters,
#     stop_loss_threshold=stop_loss_threshold
#
# )
#
# result = backtester.run()
#
# print("回测结果：")
# print("最终资金: ", result['final_cash'])
# print("交易历史: ")
# for t in result['transaction_history']:
#     print(t)
# print(backtester.account.transaction)
