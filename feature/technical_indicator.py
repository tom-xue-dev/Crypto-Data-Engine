import pickle
import sys

import numpy as np
import pandas as pd
import talib
from IC_calculator import compute_ic, compute_prediction_metrics, compute_zscore
from read_large_files import map_and_load_pkl_files, select_assets
from CUSUM_filter import generate_filter_df


def compute_aroonosc(group, timeperiod):
    group = group.copy()
    group['AROONOSC'] = talib.AROONOSC(group['high'], group['low'], timeperiod=timeperiod)
    return group


def compute_sma(group, timeperiod):
    group = group.copy()
    group['SMA'] = talib.SMA(group['close'], timeperiod=timeperiod)
    return group


def compute_ema(group, timeperiod):
    group = group.copy()
    group['EMA'] = talib.EMA(group['close'], timeperiod=timeperiod)
    return group


def compute_atr(group, timeperiod):
    group = group.copy()
    group['ATR'] = talib.ATR(group['high'], group['low'], group['close'], timeperiod=timeperiod)
    return group


def compute_admi(group, timeperiod):
    group = group.copy()
    group['ADMI'] = talib.ATR(group['high'], group['low'], group['close'], timeperiod=timeperiod)
    return group


def compute_willians(group, timeperiod):
    group = group.copy()
    group['WILLR'] = talib.ATR(group['high'], group['low'], group['close'], timeperiod=timeperiod)
    return group


def compute_stochastic(group, timeperiod):
    group = group.copy()
    group['K'], group['D'] = talib.STOCH(group['high'], group['low'], group['close'], fastk_period=timeperiod)
    return group


def compute_roc(group, timeperiod):
    group = group.copy()
    group['ROC'] = talib.ROC(group['close'], timeperiod=timeperiod)
    return group


def compute_adx(group, timeperiod):
    group = group.copy()
    group['ADX'] = talib.ADX(group['high'], group['low'], group['close'],
                             timeperiod=timeperiod)
    return group


def compute_apo(group, fast_period, slow_period):
    group = group.copy()
    group['APO'] = talib.APO(group['close'], fast_period, slow_period,
                             matype=0)
    return group


def compute_bop(group):
    group = group.copy()
    group['BOP'] = talib.BOP(group['open'], group['high'], group['low'], group['close'])
    return group


def compute_cci(group, time_period):
    group = group.copy()
    group['CCI'] = talib.CCI(group['high'], group['low'], group['close'], timeperiod=time_period) / 100
    return group


def compute_cmo(group, time_period):
    group = group.copy()
    group['CMO'] = talib.CMO(group['close'], timeperiod=time_period)
    return group


def compute_dx(group, time_period):
    group = group.copy()
    group['DX'] = talib.CCI(group['high'], group['low'], group['close'], timeperiod=time_period)
    return group


def compute_mfi(group, time_period):
    group = group.copy()
    group['MFI'] = talib.MFI(group['high'], group['low'], group['close'], group['volume'], timeperiod=time_period)
    return group


def compute_mom(group, time_period):
    group = group.copy()
    group['MOM'] = talib.MOM(group['close'], timeperiod=time_period)
    return group


def compute_rsi(group, time_period):
    group = group.copy()
    group['RSI'] = talib.RSI(group['close'], timeperiod=time_period)
    return group


def compute_amihud(group):
    group = group.copy()
    group['return'] = group['close'].pct_change()
    group['amount'] = group['close'] * group['volume']
    group['ILLIQ'] = np.abs(group['return']) / (group['amount'])
    group['ILLIQ'] = np.where(group['amount'] == 0, np.nan,
                              np.abs(group['return']) / group['amount'])
    return group


def compute_beta_regression(group: pd.DataFrame, time_period: int) -> pd.DataFrame:
    """
    针对单个资产的数据，使用过去 time_period 天（或 K 线）的最低价和最高价
    进行最小二乘回归，计算回归斜率 beta，
    beta 定义为：beta = Cov(low, high) / Var(low)

    参数：
        group : pd.DataFrame
            单个资产的数据，要求至少包含 'low' 和 'high' 列。
        time_period : int
            用于回归计算的滚动窗口大小，即使用最近 time_period 天（或 K 线）的数据。

    返回：
        pd.DataFrame:
            增加了 'beta' 列后的 DataFrame，索引与原 group 保持一致。
    """
    group = group.copy()
    # 计算滚动窗口内 'low' 与 'high' 的协方差和 'low' 的方差
    rolling_cov = group['low'].rolling(window=time_period, min_periods=time_period).cov(group['high'])
    rolling_var = group['low'].rolling(window=time_period, min_periods=time_period).var()
    # 计算 beta
    beta_raw = np.where(rolling_var == 0, np.nan, rolling_cov / rolling_var)
    group['beta'] = pd.Series(beta_raw, index=group.index).ffill()
    beta_mean = group['beta'].rolling(window=1200, min_periods=time_period).mean()
    beta_std = group['beta'].rolling(window=1200, min_periods=time_period).std()
    group['beta_zscore'] = (group['beta'] - beta_mean) / beta_std
    return group


def compute_return_skew_kurt(group: pd.DataFrame, time_period: int) -> pd.DataFrame:
    group = group.copy()
    # 计算收益率（百分比变化率）
    group['return'] = group['close'].pct_change()
    # 利用滚动窗口计算收益率的偏度和峰度，只有当窗口内数据足够时才计算（否则返回 NaN）
    group['return_skew'] = group['return'].rolling(window=time_period, min_periods=time_period).skew()
    return group


def compute_corr(group, time_period=120):
    """
    计算 DataFrame 中 'close' 列与领先成交量之间的滚动相关系数。

    参数:
      group: DataFrame，必须包含 'close' 和 'volume' 两列
      time_period: 整数，滚动窗口的周期

    返回:
      增加了两列的新 DataFrame：
        - 'leading_volume': 下一分钟的成交量（即 volume 向上平移一位）
        - 'corr': 'close' 与 'leading_volume' 在滚动窗口内计算得到的相关系数
    """
    # 复制数据，防止对原数据修改
    group = group.copy()
    group['corr'] = group['close'].rolling(window=time_period).corr(group['volume'])
    return group


def large_order_push_timeseries(df: pd.DataFrame, top_k: float = 0.3, T: int = 50):
    """
    计算“单一标的”的大单推动涨幅时序因子。

    参数:
    -------
    df : pd.DataFrame
        假设至少包含以下列：
          - 'return' : 每日收益率 (可以是简单收益率 r_t = (P_t / P_{t-1} - 1))
          - 'volume' : 每日成交金额
        index 为日期（升序，不要有重复日期）。
    top_k : float
        过去 T 天中，选取成交金额最大的 top_k 比例(0 < top_k <= 1)。例如 0.3 表示选取过去 T 天成交额最高的 30% 日期。
    T : int
        回看窗口长度，过去 T 天。

    返回:
    -------
    pd.Series
        与 df 同索引，每个日期对应一个大单推动涨幅值 (可能在最前面的 T-1 行因为缺数据而是 NaN)。
    """

    # 提取 numpy 数组以加速
    # 假定列名分别是 'return' 和 'volume'
    print(df)
    arr_ret = df['return'].values
    arr_vol = df['volume'].values

    n = len(df)
    # 用 NaN 初始化结果
    factor = np.full(shape=n, fill_value=np.nan, dtype=float)

    for t in range(T, n):
        # 1) 取过去 T 天区间： [t-T, t)
        #    如果你想包括当前 t 日自身，可以改成 [t-T+1, t+1)
        start_idx = t - T
        end_idx = t  # Python 切片到 t 不包括 t，即 [start_idx, end_idx)

        window_vol = arr_vol[start_idx:end_idx]
        window_ret = arr_ret[start_idx:end_idx]

        # 2) 根据成交额找出 top_k% 的“高成交日”
        #    如果 top_k=0.3，相当于取这些 T 天里成交额排名前 30% 的那些天
        #    (1 - top_k)*100 = 70，意味着找 volume ≥ 70分位的部分
        threshold = np.percentile(window_vol, (1 - top_k) * 100)

        # 找出满足 volume >= threshold 的索引
        # idx_set 是相对窗口内的下标
        idx_set = np.where(window_vol >= threshold)[0]

        # 3) 对这些“高成交日”的收益率做连乘，得到区间累计收益
        #    也可以改成其他聚合方式，如 simple sum / mean
        selected_rets = window_ret[idx_set]

        if len(selected_rets) > 0:
            # 连乘
            cum_ret = np.sum(selected_rets)
            # 这里可以直接作为因子值
            factor[t] = cum_ret
        else:
            # 如果没有满足条件的天数，就 NaN 或者设为 0
            factor[t] = np.nan

    # 封装成 pandas.Series，和原 df 共用 index
    factor_series = pd.Series(factor, index=df.index, name='large_order_push')
    return factor_series


if __name__ == '__main__':
    momentum_functions = talib.get_function_groups()
    print(momentum_functions)
    start = "2020-1-1"
    end = "2022-12-31"
    assets = select_assets(start_time=start, spot=True, n=20)
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")

    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x.shift(-1) - 1).droplevel(0)
    data['label'] = np.where(data['future_return']>0,1,0)
    data['return'] = data.groupby('asset')['close'].pct_change()
    # 对每个资产分组计算 AROON 指标
    time_period = 10
    print("start")
    data['factor'] = data.groupby('asset', group_keys=False).apply(lambda x: large_order_push_timeseries(x))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_sma(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_stochastic(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_admi(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_willians(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_cci(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_roc(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_ema(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_rsi(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_zscore(x, column='RSI', window=1200))
    # data['factor'] = data['corr']
    # print(data)
    data = data.dropna()
    print(data)
    print(data.columns)
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # data = generate_filter_df(data)
    ic = compute_ic(df=data, feature_column='factor', return_column='future_return')
    # ic = compute_ic(df=data, feature_column='zscore_RSI', return_column='future_return')
    print("IC_MEAN:", np.mean(ic), "IR", np.mean(ic) / np.std(ic))
