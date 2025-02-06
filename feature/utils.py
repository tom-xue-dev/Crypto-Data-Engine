import pandas as pd
import numpy as np
from pandas.core.algorithms import rank as rk


### Initial Operations
def returns(df):
    return df.groupby(level='asset', group_keys=False).apply(lambda x: x-x.shift(-1)) - 1



def vwap(df):
    """
    计算累计 VWAP（Volume Weighted Average Price）

    对于每个资产，计算截止到每个时间点的累计 VWAP，即：
      累计 VWAP = 累计 (price * volume) / 累计 volume

    参数:
      df: 包含多级索引的 DataFrame，索引第一级为 'time'，
          第二级为 'asset'，并且至少包含 'close' 和 'volume' 两列。

    返回:
      一个 Series，索引与 df 相同，表示每个 (time, asset) 的累计 VWAP。
    """
    # 复制数据并确保按照时间排序（对每个资产都按时间排序）
    df = df.copy().sort_index(level='time')

    # 计算每笔交易的价格 * 成交量
    df['price_volume'] = (df.close + df.high + df.low) / 3 * df['volume']

    # 对每个资产计算累积求和
    df['cum_price_volume'] = df.groupby(level='asset')['price_volume'].cumsum()
    df['cum_volume'] = df.groupby(level='asset')['volume'].cumsum()

    # 计算累计 VWAP，注意处理 cum_volume 为 0 的情况
    df['vwap'] = df['cum_price_volume'] / df['cum_volume']
    df.loc[df['cum_volume'] == 0, 'vwap'] = np.nan

    # 返回累计 VWAP 列
    # return df['vwap']
    return (df.close + df.high + df.low) / 3


def adv(df, d):
    """
    adv{d} = average daily dollar volume for the past d days 
    """
    return df.volume.rolling(d).mean()


###
def rank(df):
    """
    Cross-sectional percentile rank.

    :param df:
    :return: 
    """
    df_rank = df.groupby('time', group_keys=False).apply(
        lambda x: pd.Series(rk(x.values, method='max', pct=True), index=x.index))
    return df_rank


def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.groupby(level='asset', group_keys=False).rolling(window).std().droplevel(0)


def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.groupby(level='asset', group_keys=False).apply(lambda x: x.mul(k).div(np.abs(x).sum()))


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply.
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.groupby(level='asset', group_keys=False).rolling(window).apply(rolling_prod)


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    weights = np.array(range(1, period + 1))
    sum_weights = np.sum(weights)
    return df.groupby(level='asset', group_keys=False).rolling(period).apply(
        lambda x: np.sum(weights * x) / sum_weights).droplevel(0)


def delta(df, d):
    """
    today’s value of x minus the value of x d days ago
    """
    df = df.groupby('asset', group_keys=False).apply(lambda x: x - x.shift(d))
    return df


def corr(x, y, d):
    """
    time-serial correlation of x and y for the past d days 
    """
    df = pd.concat([x, y], axis=1, keys=["df1", "df2"])
    df = df.dropna()
    rolling_corr_all = df.groupby(level="asset", group_keys=False).apply(
        lambda x: x["df1"].rolling(window=d).corr(x["df2"])
    )
    return rolling_corr_all


def cov(x, y, d):
    """
    time-serial covariance of x and y for the past d days 
    """
    df = pd.concat([x, y], axis=1, keys=["df1", "df2"])
    rolling_cov_all = df.groupby(level="asset", group_keys=False).apply(
        lambda x: x["df1"].rolling(window=d).cov(x["df2"])
    )
    return rolling_cov_all


def delay(df, d):
    """
    value of x d days ago
    """
    return df.groupby(level="asset", group_keys=False).shift(d)


### Time-Series Operations
def ts_max(df, d=10):
    """
    The rolling max over the last d days. 

    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.groupby(level="asset", group_keys=False).rolling(d).max().droplevel(0)


def ts_min(df, d=10):
    """
    The rolling min over the last d days.

    :param df: data frame containing prices
    :param d: number of days to look back (rolling window)
    :return: Pandas series
    """
    return df.groupby(level="asset", group_keys=False).rolling(d).min().droplevel(0)


def ts_argmax(df, d):
    """
    Gets the day, ts_max(x, d) occured on.

    :param df: dataframe
    :param d: number of days to look back (rolling window)
    :return: Pandas Series
    """
    return df.groupby(level="asset", group_keys=False).rolling(d).apply(np.argmax).add(1).droplevel(0)


def ts_argmin(df, d):
    """
    Gets the day, ts_min(x, d) occured on.

    :param df: dataframe
    :param d: number of days to look back (rolling window)
    :return: Pandas Series
    """
    return df.groupby(level="asset", group_keys=False).rolling(d).apply(np.argmin).add(1).droplevel(0)


def ts_rank(df, d):
    """
    time-series rank in the past d days
    
    :param df: dataframe
    :param d: number of days to look back (rolling window)
    :return: Pandas Series
    """
    return df.groupby(level="asset", group_keys=False).rolling(d).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).droplevel(0)


def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum for MultiIndex DataFrame.
    :param df: a pandas DataFrame with MultiIndex (date, asset).
    :param window: the rolling window size.
    :return: a pandas DataFrame with the rolling sum over the past 'window' days per asset.
    """
    return df.groupby(level='asset', group_keys=False).rolling(window).sum().droplevel(0)
