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
    dataset.loc[dataset[short_ma_col] > dataset[long_ma_col], 'signal'] = -1  # 短期均线高于长期均线 -> 买入
    dataset.loc[dataset[short_ma_col] < dataset[long_ma_col], 'signal'] = 1  # 短期均线低于长期均线 -> 卖出

    return dataset
