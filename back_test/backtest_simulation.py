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
        if signal == 1 and cash > 0:
            price *= 1.001 # 增加千分之1的手续费，相当于提高千分之一的价格
            position = cash / price  # 买入的数量
            cash = 0
            buy_price = price
            print(f"Buy at {price:.2f}, position: {position:.6f}")
        # 卖出信号：卖出所有持仓，转换为现金
        elif signal == -1 and position > 0:
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

df = get_btc_data("15m")
df = df.sort_index(ascending=False)#重新排序
df = strategy.double_moving_average_strategy(3, 15, df)
data = backtest(df)
print(data)
# plot_with_signals(df, length=2000)
plot_portfolio_value(df,data,length=len(df))
