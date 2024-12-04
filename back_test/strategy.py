import pandas as pd

from Account import Account
from get_btc_info import get_btc_data


def calculate_MA(dataset, n):
    """
    计算 n 日均线并将其添加到 dataset 的新列中。

    :param dataset: pd.DataFrame, 数据集，必须包含 'close' 列
    :param n: int, 均线的天数
    :return: pd.DataFrame, 添加了均线列后的数据集
    """
    # 检查数据集是否包含 'close' 列
    if 'close' not in dataset.columns:
        raise ValueError("The dataset must contain a 'close' column.")

    # 计算 n 日均线
    column_name = f'MA{n}'
    dataset[column_name] = dataset['close'].rolling(window=n).mean()
    return dataset


def double_moving_average_strategy(short, long, dataset):
    """
    实现双均线策略，生成买入、卖出和持有信号。

    :param short: int, 短期均线天数
    :param long: int, 长期均线天数
    :param dataset: pd.DataFrame, 数据集，必须包含 'close' 列
    :return: pd.DataFrame, 包含策略信号的原始数据集
    """
    # 计算短期和长期均线
    dataset = calculate_MA(dataset, short)
    dataset = calculate_MA(dataset, long)

    # 列名动态生成
    short_ma_col = f'MA{short}'
    long_ma_col = f'MA{long}'

    # 生成交易信号
    dataset['signal'] = 0  # 初始化信号为 0（保持）
    dataset.loc[dataset[short_ma_col] > dataset[long_ma_col], 'signal'] = 1  # 短期均线高于长期均线 -> 买入
    dataset.loc[dataset[short_ma_col] < dataset[long_ma_col], 'signal'] = -1  # 短期均线低于长期均线 -> 卖出
    return dataset


def arbitrage_trading_trategy(dataset1,dataset2):
    """

    :param dataset1:
    :param dataset2:
    :return:
    """
    dataset1['time'] = pd.to_datetime(dataset1['time'])
    dataset2['time'] = pd.to_datetime(dataset2['time'])

    # 合并两个 DataFrame，按 'time' 对齐
    merged = pd.merge(dataset1, dataset2, on='time', suffixes=('_df1', '_df2'))
    # 计算差值
    merged['open_diff'] = merged['open_df1'] - merged['open_df2']
    merged['close_diff'] = merged['close_df1'] - merged['close_df2']
    merged['high_diff'] = merged['high_df1'] - merged['high_df2']
    merged['low_diff'] = merged['low_df1'] - merged['low_df2']
    # initialize signal column
    merged['signal'] = 0
    # Condition 1: Arbitrage opportunity for opening long on A and short on B
    merged.loc[
        (merged['open_diff'] > 0) & (merged['open_diff'] > merged['close_df1'] * 0.001),
        'signal'
    ] = -1

    # Condition 2: Arbitrage opportunity for opening short on A and long on B
    merged.loc[
        (merged['open_diff'] < 0) & (merged['open_diff'].abs() > merged['close_df1'] * 0.001),
        'signal'
    ] = 1
    # condition 3,平仓
    merged.loc[merged['open_diff'].abs() < 2, 'signal'] = 2
    return merged
# df = get_btc_data("15m")
# df = calculate_MA(df,20)
# print(df.tail(10))