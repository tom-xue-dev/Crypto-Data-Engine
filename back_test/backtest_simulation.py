from datetime import datetime

import numpy as np
import pandas as pd

from Account import Account
from get_btc_info import get_btc_data
import strategy
import matplotlib.pyplot as plt


def backtest(dataset, initial_cash=10000, leverage=1, hourly_rate=0.00180314):
    """
    根据交易信号进行回测，计算账户的资产变化和策略表现，支持杠杆交易及借贷利息。

    :param dataset: pd.DataFrame, 数据集，必须包含 'signal' 和 'close' 列
    :param initial_cash: float, 初始资金
    :param leverage: float, 杠杆倍数，默认为1（不使用杠杆）
    :param hourly_rate: float, 借贷的小时利率
    :return: dict, 包含回测结果的字典
    """
    if 'signal' not in dataset.columns or 'close' not in dataset.columns:
        raise ValueError("The dataset must contain 'signal' and 'close' columns.")

    cash = initial_cash  # 初始现金
    position = 0  # 初始持仓
    portfolio_value = []  # 每个时间点的总资产价值
    date_time = []
    buy_price = 0  # 记录买入价格
    borrow_amount = 0  # 记录借入金额
    borrow_start_time = None  # 记录借款时间

    for i in range(len(dataset)):
        date_time.append(dataset['time'].iloc[i])
        price = dataset['close'].iloc[i]
        signal = dataset['signal'].iloc[i]

        # 买入信号：用所有现金（加杠杆）买入资产
        if signal == -1 and cash > 0:
            price *= 1.001  # 增加千分之1的手续费
            borrow_amount = cash * (leverage - 1)  # 计算借款金额
            position = (cash + borrow_amount) / price  # 计算总持仓
            cash = 0  # 持仓后现金归零
            buy_price = price
            borrow_start_time = dataset['time'].iloc[i]
            print(f"Buy at {price:.2f}, position: {position:.6f}, leverage: {leverage}, borrowed: {borrow_amount:.2f}")

        # 卖出信号：卖出所有持仓，转换为现金
        elif signal == 1 and position > 0:
            price *= 0.999  # 增加千分之一手续费
            cash = position * price  # 卖出后获得的现金
            position = 0

            # 计算借贷利息
            hours_held = (dataset['time'].iloc[i] - borrow_start_time).total_seconds() / 3600
            interest = borrow_amount * hourly_rate * hours_held
            cash -= interest  # 扣除借贷利息
            borrow_amount = 0
            borrow_start_time = None

            print(f"Sell at {price:.2f}, cash: {cash:.2f}, interest paid: {interest:.2f}")

        # 记录当前资产价值
        total_value = cash + position * price
        portfolio_value.append(total_value)

        # 检测爆仓
        if total_value <= 0:
            print(f"Liquidated at time {date_time[-1]} with price {price:.2f}")
            break

    # 计算最终结果
    final_value = cash + position * dataset['close'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    result_frame = pd.DataFrame({'value': portfolio_value, 'time': date_time})
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


def back_test(strategy_results, initial_capital=10000, position_size=1, leverage=5, hourly_rate=0.00180314):
    """
    回测套利策略的收益表现，支持杠杆交易及借贷利率。

    :param strategy_results: 包含交易信号的 DataFrame，需包含 'time', 'signal', 'open_df1', 'open_df2' 列。
    :param initial_capital: float, 初始资金 (默认 $10,000)。
    :param position_size: float, 每次交易的基础仓位大小 (单位: BTC 或其他资产)。
    :param leverage: float, 杠杆倍数，默认为1（不使用杠杆）。
    :param hourly_rate: float, 借贷的小时利率。
    :return: tuple (DataFrame, float)，回测结果和最终资金。
    """
    # 初始化资金、状态和交易记录
    capital = initial_capital
    open_position = None  # 记录当前持仓 ('long' or 'short')
    entry_price_a = entry_price_b = 0  # 记录开仓价格
    borrow_amount = 0  # 记录借款金额
    borrow_start_time = None  # 记录借款开始时间
    results = []  # 存储回测的每一步结果

    for index, row in strategy_results.iterrows():
        # 开仓信号处理
        if row['signal'] == 1 and open_position is None:
            # 在 A 开多，B 开空
            open_position = 'long'
            entry_price_a = row['open_df1']
            entry_price_b = row['open_df2']
            borrow_amount = capital * (leverage - 1)  # 计算借款金额
            position_size *= leverage  # 调整仓位大小
            borrow_start_time = row['time']  # 记录借款开始时间
            print(f"Opened long A and short B with leverage: {leverage}, borrowed: {borrow_amount:.2f}")

        elif row['signal'] == -1 and open_position is None:
            # 在 A 开空，B 开多
            open_position = 'short'
            entry_price_a = row['open_df1']
            entry_price_b = row['open_df2']
            borrow_amount = capital * (leverage - 1)  # 计算借款金额
            position_size *= leverage  # 调整仓位大小
            borrow_start_time = row['time']  # 记录借款开始时间
            print(f"Opened short A and long B with leverage: {leverage}, borrowed: {borrow_amount:.2f}")

        # 平仓信号处理
        elif row['signal'] == 2 and open_position is not None:
            # 计算持仓时间
            hours_held = (row['time'] - borrow_start_time).total_seconds() / 3600
            interest = borrow_amount * hourly_rate * hours_held  # 计算借贷利息

            if open_position == 'long':
                # 平仓：A 平多，B 平空
                profit_a = (row['close_df1'] - entry_price_a) * position_size
                profit_b = (entry_price_b - row['close_df2']) * position_size
            elif open_position == 'short':
                # 平仓：A 平空，B 平多
                profit_a = (entry_price_a - row['close_df1']) * position_size
                profit_b = (row['close_df2'] - entry_price_b) * position_size
            else:
                profit_a = profit_b = 0

            # 总利润扣除借贷利息
            total_profit = profit_a + profit_b - interest
            capital += total_profit
            borrow_amount = 0  # 重置借款金额
            borrow_start_time = None  # 重置借款时间

            if capital <= 0:
                print(f"Liquidated at time {row['time']} due to insufficient capital.")
                break

            print(f"Closed position at time {row['time']} with profit: {total_profit:.2f}, interest: {interest:.2f}, capital: {capital:.2f}")

            # 记录结果
            results.append({
                'time': row['time'],
                'signal': 2,
                'profit_a': profit_a,
                'profit_b': profit_b,
                'total_profit': total_profit,
                'interest': interest,
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
