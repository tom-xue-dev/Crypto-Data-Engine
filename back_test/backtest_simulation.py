import math
from datetime import datetime

import numpy as np
import pandas as pd

from Account import Account
from get_btc_info import get_btc_data
import strategy
import matplotlib.pyplot as plt
import matplotlib.dates as dates


def backtest(strategy_results, initial_capital=10000, position_size=0.0001, leverage=1, hourly_rate=0.00180314):
    """
    回测套利策略的收益表现，支持杠杆交易及借贷利率，且保留收益在各自交易所。

    :param strategy_results: 包含交易信号的 DataFrame，需包含 'time', 'open', 'close','high','low'列
    :param initial_capital: float, 初始资金 (默认 $10,000)。
    :param position_size: float, 每次交易的基础仓位大小 (单位: BTC 或其他资产)。
    :param leverage: float, 杠杆倍数，默认为1（不使用杠杆）。
    :param hourly_rate: float, 借贷的小时利率。
    :return: tuple (DataFrame, float, float)，回测结果和两个交易所的最终资金。
    """
    # 目前只支持现货交易
    total_capital = initial_capital
    account = Account(1000,['AAPL', 'GOOG', 'AMZN'])
    for daily_df in strategy_results:
        for index,row in daily_df.iterrows():
            if row['signal'] == 1:
                receive_amount = account.cash / row['close']
                receive_amount = math.floor(receive_amount/position_size) * position_size
                if receive_amount < position_size:
                    print("insufficient cash")
                    continue
                sell_amount = receive_amount * row['close']
                print(f"try to buy{row['asset']},ask = {receive_amount},total_cost = {sell_amount}")
                account.buy(row['time'],row['asset'],receive_amount,"USD",sell_amount)
            elif row['signal'] == 0:
                current_holding = account.holdings
                if row['asset'] not in current_holding or current_holding[row['asset']] < position_size:
                    print("Insufficient security to sell.")
                    continue

                sell_amount = current_holding[row['asset']]
                receive_amount = sell_amount * row['close']
                print(f"Trying to sell {sell_amount} of {row['asset']}, receive = {receive_amount}")
                account.sell(row['time'],row['asset'], sell_amount, "USD", receive_amount)
    print(account.get_transaction_history())




if __name__ == "__main__":
    print("backtest start")
    data_day1 = pd.DataFrame({
        'time': ['2024-01-01'] * 3,
        'asset': ['AAPL', 'GOOG', 'AMZN'],
        'open': [150, 2800, 3450],
        'high': [152, 2825, 3480],
        'close': [151, 2810, 3460]
    })

    data_day2 = pd.DataFrame({
        'time': ['2024-01-02'] * 3,
        'asset': ['AAPL', 'GOOG', 'AMZN'],
        'open': [152, 2825, 3480],
        'high': [153, 2835, 3490],
        'close': [152, 2830, 3485]
    })
    data_day3 = pd.DataFrame({
        'time': ['2024-01-03'] * 3,
        'asset': ['AAPL', 'GOOG', 'AMZN'],
        'open': [152, 2825, 3480],
        'high': [153, 2835, 3490],
        'close': [152, 2830, 3485]
    })
    dataset = [data_day1, data_day2, data_day3]
    assets = ['AAPL', 'GOOG', 'AMZN']
    # 打印 DataFrame
    strategy = strategy.MovingAverageStrategy(dataset, assets, 2)

    strategy.calculate_MA(3)
    signal_data = strategy.generate_signal()
    backtest(signal_data,1000,0.1)
