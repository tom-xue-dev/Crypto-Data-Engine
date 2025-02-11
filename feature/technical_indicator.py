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
    group['CCI'] = talib.CCI(group['high'], group['low'], group['close'], timeperiod=time_period)
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
    group['beta'] = rolling_cov / rolling_var
    return group



if __name__ == '__main__':
    momentum_functions = talib.get_function_groups()['Momentum Indicators']
    print(momentum_functions)
    start = "2023-1-1"
    end = "2023-12-31"
    assets = select_assets(start_time=start, spot=True, n=50)
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")

    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)

    # 对每个资产分组计算 AROON 指标
    fast_period = 12  # 快速EMA的周期
    slow_period = 26  # 慢速EMA的周期
    time_period = 12

    print(data)

    data = data.groupby('asset', group_keys=False).apply(lambda x: compute_beta_regression(x, time_period))
    # data = data.groupby('asset', group_keys=False).apply(lambda x: compute_zscore(x, column='RSI', window=1200))
    data['factor'] = data['beta']
    # print(data)
    data = generate_filter_df(data)
    ic = compute_ic(df=data, feature_column='factor', return_column='future_return')
    #ic = compute_ic(df=data, feature_column='zscore_RSI', return_column='future_return')
    print("IC_MEAN:", np.mean(ic), "IR", np.mean(ic) / np.std(ic))
