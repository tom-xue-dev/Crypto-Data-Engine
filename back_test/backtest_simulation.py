from datetime import datetime

import numpy as np
import pandas as pd

from Account import Account
from get_btc_info import get_btc_data
import strategy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def backtest(dataset, initial_cash=10000, leverage=5, hourly_rate=0.00180314):
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
    initial_position_size = 0

    for i in range(len(dataset)):
        date_time.append(dataset['time'].iloc[i])
        price = dataset['close'].iloc[i]
        signal = dataset['signal'].iloc[i]

        # 买入信号：用所有现金（加杠杆）买入资产
        if signal == -1 and cash > 0:
            price *= 1.001  # 增加千分之1的手续费
            borrow_amount = cash * (leverage - 1)  # 计算借款金额
            initial_position_size = (cash + borrow_amount) / price  # 初始持仓大小
            position = initial_position_size  # 计算总持仓
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
            initial_position_size = 0  # 重置初始持仓大小

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
    return result_frame, results





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

    scaled_close = data_subset['close'] / data_subset['close'].iloc[0]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(data_subset['time'], scaled_close, data_subset['close'], label='Close Price')

    buy_signals = data_subset[data_subset['signal'] == 1]
    sell_signals = data_subset[data_subset['signal'] == -1]

    # add buy and sell signal in the picture
    ax.scatter(buy_signals['time'], buy_signals['close'] / data_subset['close'].iloc[0], color='green', marker='^', label='Buy Signal', s=100)
    ax.scatter(sell_signals['time'], sell_signals['close'] / data_subset['close'].iloc[0], color='red', marker='v', label='Sell Signal', s=100)
    # set labels and the x axes.
    ax.xaxis.set_ticks(np.arange(0, length, 50))
    ax.set_xlabel("Time")
    ax.set_ylabel("Close Price")
    ax.set_title("Close Price with Buy/Sell Signals")

    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_portfolio_value(original_data, portfolio_data, length=200):
    """
    绘制投资组合价值与基准的对比曲线，将值缩放到从 1 开始。

    参数：
        original_data: pd.DataFrame
            包含基准数据（'time' 和 'close' 列）的数据集。
        portfolio_data: pd.DataFrame
            包含投资组合数据（'time' 和 'value' 列）的数据集。
        length: int
            显示的最大数据点数量，默认为 200。
    """
    # 确保 length 不超过数据集总长度
    length = min(len(portfolio_data), length)
    pdata_subset = portfolio_data.iloc[:length]
    odata_subset = original_data.iloc[:length]

    # 缩放投资组合价值和基准值
    scaled_portfolio_value = pdata_subset['value'] / pdata_subset['value'].iloc[0]
    scaled_benchmark = odata_subset['close'] / odata_subset['close'].iloc[0]

    # 创建图表
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(pdata_subset['time'], scaled_portfolio_value, label='投资组合价值（缩放）', c="blue", linewidth=2)
    ax.plot(odata_subset['time'], scaled_benchmark, label='基准（缩放）', c="red", linestyle='--', linewidth=2)

    # 设置轴标签、标题和图例
    ax.set_xlabel("时间")
    ax.set_ylabel("价值（缩放）")
    ax.set_title("投资组合价值与基准对比（缩放）")
    ax.legend()

    # 动态格式化时间轴
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def back_test(strategy_results, initial_capital=10000, position_size=1, leverage=5, hourly_rate=0.00180314):
    """
    回测套利策略的收益表现，支持杠杆交易及借贷利率。

    :param strategy_results: 包含交易信号的 DataFrame，需包含 'time', 'signal', 'open_df1', 'open_df2', 'close_df1', 'close_df2' 列。
    :param initial_capital: float, 初始资金 (默认 $10,000)。
    :param position_size: float, 每次交易的基础仓位大小 (单位: BTC 或其他资产)。
    :param leverage: float, 杠杆倍数，默认为1（不使用杠杆）。
    :param hourly_rate: float, 借贷的小时利率。
    :return: tuple (DataFrame, float)，回测结果和最终资金。
    """
    capital_per_exchange = initial_capital / 2  # 每个交易所初始资金的一半
    open_position = None  # 记录当前持仓 ('long' or 'short')
    entry_price_a = entry_price_b = 0  # 记录开仓价格
    borrow_amount_a = borrow_amount_b = 0  # 记录每个交易所的借款金额
    borrow_start_time = None  # 记录借款开始时间
    results = []  # 存储回测的每一步结果

    for index, row in strategy_results.iterrows():
        # 开仓信号处理
        if row['signal'] == 1 and open_position is None:
            open_position = 'long'
            entry_price_a = row['open_df1']
            entry_price_b = row['open_df2']
            borrow_amount_a = capital_per_exchange * (leverage - 1)  # A交易所借款金额
            borrow_amount_b = capital_per_exchange * (leverage - 1)  # B交易所借款金额
            position_size *= leverage  # 调整仓位大小
            borrow_start_time = row['time']  # 记录借款开始时间

        elif row['signal'] == -1 and open_position is None:
            open_position = 'short'
            entry_price_a = row['open_df1']
            entry_price_b = row['open_df2']
            borrow_amount_a = capital_per_exchange * (leverage - 1)
            borrow_amount_b = capital_per_exchange * (leverage - 1)
            position_size *= leverage
            borrow_start_time = row['time']

        # 平仓信号处理
        elif row['signal'] == 2 and open_position is not None:
            hours_held = (row['time'] - borrow_start_time).total_seconds() / 3600
            interest_a = borrow_amount_a * hourly_rate * hours_held
            interest_b = borrow_amount_b * hourly_rate * hours_held

            if open_position == 'long':
                profit_a = (row['close_df1'] - entry_price_a) * position_size / 2
                profit_b = (entry_price_b - row['close_df2']) * position_size / 2
            elif open_position == 'short':
                profit_a = (entry_price_a - row['close_df1']) * position_size / 2
                profit_b = (row['close_df2'] - entry_price_b) * position_size / 2
            else:
                profit_a = profit_b = 0

            transaction_fee_rate = 0.001  # 交易费用比例

            transaction_fee_a = (entry_price_a + row['close_df1']) * position_size / 2 * transaction_fee_rate
            transaction_fee_b = (entry_price_b + row['close_df2']) * position_size / 2 * transaction_fee_rate
            total_transaction_fee = transaction_fee_a + transaction_fee_b
            total_profit = profit_a + profit_b - interest_a - interest_b - total_transaction_fee
            capital_per_exchange += total_profit / 2  # 将收益平分到两个交易所资金中
            borrow_amount_a = borrow_amount_b = 0
            borrow_start_time = None
            position_size = 1  # 重置仓位大小

            if capital_per_exchange <= 1e-6:
                print(f"Liquidated at time {row['time']} due to insufficient capital in one exchange.")
                break

            results.append({
                'time': row['time'],
                'signal': 2,
                'profit_a': profit_a,
                'profit_b': profit_b,
                'total_profit': total_profit,
                'interest_a': interest_a,
                'interest_b': interest_b,
                'capital_a': capital_per_exchange,
                'capital_b': capital_per_exchange
            })

            open_position = None
            entry_price_a = entry_price_b = 0

    # 转换结果为 DataFrame
    results_df = pd.DataFrame(results)

    return results_df, capital_per_exchange * 2  # 返回两个交易所总资金的和





start_date = datetime(2023, 11, 1)
end_date = datetime(2024, 11, 1)
timeframe = '1m'
exchange_name1 = 'binance'
exchange_name2 = 'bybit'
crypto_type = 'BTCUSDT'
binance_data = get_btc_data(start_date, end_date, timeframe, exchange_name1, crypto_type)
bybit_data = get_btc_data(start_date, end_date, timeframe, exchange_name2, crypto_type)
df = strategy.arbitrage_trading_trategy(binance_data,bybit_data)
print(df)
print(back_test(df))

