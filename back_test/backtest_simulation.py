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
    buy_price = 0  # 记录买入价格（用于计算持仓收益）
    dataset = dataset.sort_index(ascending=False)
    for i in range(len(dataset)):
        price = dataset['close'].iloc[i]*1.001
        signal = dataset['signal'].iloc[i]

        # 买入信号：用所有现金买入资产
        if signal == 1 and cash > 0:
            position = cash / price  # 买入的数量
            cash = 0
            buy_price = price
            print(f"Buy at {price:.2f}, position: {position:.6f}")
        # 卖出信号：卖出所有持仓，转换为现金
        elif signal == -1 and position > 0:
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

    # 返回回测结果
    results = {
        "Initial Cash": initial_cash,
        "Final Portfolio Value": final_value,
        "Total Return (%)": total_return,
    }
    return results


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

    # 截取数据范围
    data_subset = dataset.iloc[:length]

    # 设置图表大小（限制最大屏幕显示）
    plt.figure(figsize=(16, 8))  # 宽16英寸，高8英寸

    # 绘制 close_price，横轴为 time
    plt.plot(data_subset['time'], data_subset['close'], label='Close Price')

    # 获取买入和卖出信号的索引
    buy_signals = data_subset[data_subset['signal'] == 1]
    sell_signals = data_subset[data_subset['signal'] == -1]

    # 在图上添加买入信号（绿色点）
    plt.scatter(buy_signals['time'], buy_signals['close'], color='green', marker='^', label='Buy Signal', s=100)

    # 在图上添加卖出信号（红色点）
    plt.scatter(sell_signals['time'], sell_signals['close'], color='red', marker='v', label='Sell Signal', s=100)

    # 添加轴标签和标题
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.title("Close Price with Buy/Sell Signals")

    # 添加图例
    plt.legend()

    # 显示图形
    plt.tight_layout()  # 优化布局，防止文字重叠
    plt.show()

df = get_btc_data("15m")
df = strategy.double_moving_average_strategy(3,15,df)
backtest(df)
plot_with_signals(df, length=2000)