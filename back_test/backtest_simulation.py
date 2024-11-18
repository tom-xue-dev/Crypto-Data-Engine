from datetime import datetime

import numpy as np
import pandas as pd

from Account import Account
from get_btc_info import get_btc_data
import strategy
import matplotlib.pyplot as plt


def backtest(dataset, initial_cash=10000):
    """
    根据交易信号进行回测，计算账户的资产变化和策略表现。

    :param dataset: pd.DataFrame, 数据集，必须包含 'signal' 和 'close' 列
    :param initial_cash: float, 初始资金
    :return: dict, 包含回测结果的字典
    """
    if 'signal' not in dataset.columns or 'close' not in dataset.columns:
        raise ValueError("The dataset must contain 'signal' and 'close' columns.")

    cash = initial_cash  # 初始现金
    position = 0  # 初始持仓
    portfolio_value = []  # 每个时间点的总资产价值
    date_time = []
    buy_price = 0  # 记录买入价格（用于计算持仓收益）

    for i in range(len(dataset)):
        date_time.append(dataset['time'].iloc[i])
        price = dataset['close'].iloc[i]
        signal = dataset['signal'].iloc[i]

        # 买入信号：用所有现金买入资产
        if signal == -1 and cash > 0:
            price *= 1.001 # 增加千分之1的手续费，相当于提高千分之一的价格
            position = cash / price  # 买入的数量
            cash = 0
            buy_price = price
            print(f"Buy at {price:.2f}, position: {position:.6f}")
        # 卖出信号：卖出所有持仓，转换为现金
        elif signal == 1 and position > 0:
            price = price * 0.999 #增加千分之一手续费，相当于减少千分之一价格
            cash = position * price  # 卖出的现金
            position = 0
            print(f"Sell at {price:.2f}, cash: {cash:.2f}")

        # 记录当前资产价值
        total_value = cash + position * price
        portfolio_value.append(total_value)

    # 计算最终结果
    final_value = cash + position * dataset['close'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    # max_drawdown = max(
    #     (max(portfolio_value) - min(portfolio_value)) / max(portfolio_value) * 100
    #     for i in range(1, len(portfolio_value))
    # )
    result_frame = pd.DataFrame({'value': portfolio_value,'time':date_time})
    # 返回回测结果
    results = {
        "Initial Cash": initial_cash,
        "Final Portfolio Value": final_value,
        "Total Return (%)": total_return,
    }
    return result_frame


def plot_with_signals(dataset, length=200):
    """
    绘制 close_price，并标记买入和卖出信号，横轴为 time。
    Parameters:
        dataset: pd.DataFrame
            包含 'time'、'close' 和 'signal' 列的数据集。
        length: int
            显示的图表数据点数量。
    """
    # 确保 length 不超过数据集的长度
    length = min(length, len(dataset))
    # create the plot
    data_subset = dataset.iloc[:length]
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(data_subset['time'], data_subset['close'], label='Close Price')

    buy_signals = data_subset[data_subset['signal'] == 1]
    sell_signals = data_subset[data_subset['signal'] == -1]

    # add buy and sell signal in the picture
    ax.scatter(buy_signals['time'], buy_signals['close'], color='green', marker='^', label='Buy Signal', s=100)
    ax.scatter(sell_signals['time'], sell_signals['close'], color='red', marker='v', label='Sell Signal', s=100)
    # set labels and the x axes.
    ax.xaxis.set_ticks(np.arange(0, length, 50))
    ax.set_xlabel("Time")
    ax.set_ylabel("Close Price")
    ax.set_title("Close Price with Buy/Sell Signals")

    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_portfolio_value(original_data,portfolio_data,length=200):
    """
    plot the portfolio value with benchmark,x-axis is time and y-axis is the value
    :param dataset: the data that need to be plotted, include time and value column
    :return: None
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    length = min(len(portfolio_data),length)
    pdata_subset = portfolio_data.iloc[:length]
    odata_subset = original_data.iloc[:length]
    ax.plot(pdata_subset['time'], pdata_subset['value'], label='Portfolio Value', c="blue")
    ax.plot(pdata_subset['time'], odata_subset['close'], label='benchmark', c="red")
    ax.xaxis.set_ticks(np.arange(0, length, 500))
    plt.show()


def back_test(strategy_results, initial_capital=10000, position_size=1):
    """
    回测套利策略的收益表现，增加对平仓信号的处理。

    :param strategy_results: 包含交易信号的 DataFrame，需包含 'time', 'signal', 'open_df1', 'open_df2' 列。
    :param initial_capital: 初始资金 (默认 $10,000)。
    :param position_size: 每次交易的仓位大小 (单位: BTC 或其他资产)。
    :return: 回测结果 DataFrame 和最终资金。
    """
    # 初始化资金、状态和交易记录
    capital = initial_capital
    open_position = None  # 记录当前持仓 ('long' or 'short')
    entry_price_a = entry_price_b = 0  # 记录开仓价格
    results = []  # 存储回测的每一步结果

    for index, row in strategy_results.iterrows():
        # 处理信号逻辑
        if row['signal'] == 1 and open_position is None:
            # 开仓信号：在 A 开多，B 开空
            open_position = 'long'
            entry_price_a = row['open_df1']
            entry_price_b = row['open_df2']
            print(f"long A short B,a_price is {entry_price_a},b_price is {entry_price_b}")
        elif row['signal'] == -1 and open_position is None:
            # 开仓信号：在 A 开空，B 开多
            open_position = 'short'
            entry_price_a = row['open_df1']
            entry_price_b = row['open_df2']
            print(f"short A long B,a_price is {entry_price_a},b_price is {entry_price_b}")
        elif row['signal'] == 2 and open_position is not None:
            # 平仓信号
            if open_position == 'long':
                # 平仓：A 平多，B 平空
                profit_a = (row['close_df1'] - entry_price_a) * position_size - row['close_df1'] * 0.001 - entry_price_a * 0.001
                profit_b = (entry_price_b - row['close_df2']) * position_size - row['close_df2'] * 0.001 - entry_price_b * 0.001
            elif open_position == 'short':
                # 平仓：A 平空，B 平多
                profit_a = (entry_price_a - row['close_df1']) * position_size - row['close_df1'] * 0.001 - entry_price_a * 0.001
                profit_b = (row['close_df2'] - entry_price_b) * position_size - row['close_df2'] * 0.001 - entry_price_b * 0.001
            else:
                profit_a = profit_b = 0

            # 计算总利润并更新资金
            total_profit = profit_a + profit_b
            capital += total_profit

            # 记录平仓结果
            results.append({
                'time': row['time'],
                'signal': 2,
                'profit_a': profit_a,
                'profit_b': profit_b,
                'total_profit': total_profit,
                'capital': capital
            })

            # 重置持仓状态
            open_position = None
            entry_price_a = entry_price_b = 0

    # 转换结果为 DataFrame
    results_df = pd.DataFrame(results)

    return results_df, capital

start_date = datetime(2021, 8, 1)
end_date = datetime(2022, 7, 1)
timeframe = '1m'
exchange_name1 = 'binance_btc'
exchange_name2 = 'bybit_btc'
binance_data = get_btc_data(start_date, end_date, timeframe, exchange_name1)
bybit_data = get_btc_data(start_date, end_date, timeframe, exchange_name2)
df = strategy.arbitrage_trading_trategy(binance_data,bybit_data)
print(df)
print(back_test(df))
