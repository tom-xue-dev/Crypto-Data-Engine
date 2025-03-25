import talib

import utils as u
import pandas as pd
import numpy as np


def compute_zscore(group: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """
    对指定的 DataFrame 分组数据，计算指定列的滚动 z-score。

    参数：
        group : pd.DataFrame
            单个资产或单个组的数据，要求包含需要计算 z-score 的列。
        column : str
            要计算 z-score 的列名称，例如 'ADX'。
        window : int
            滚动窗口大小，即用来计算均值和标准差的周期数。

    返回：
        pd.DataFrame
            增加了 z-score 列后的 DataFrame，新增的列名格式为 'zscore_{column}'。
    """
    group = group.copy()
    # 计算滚动均值和标准差，min_periods 设为 window 保证只有足够数据时才计算，否则为 NaN
    rolling_mean = group[column].rolling(window=window, min_periods=window).mean()
    rolling_std = group[column].rolling(window=window, min_periods=window).std()
    # 计算 z-score
    group[f"zscore_{column}"] = (group[column] - rolling_mean) / rolling_std
    return group


def alpha1(df):
    """
    Alpha#1
    (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)

    :param df: dataframe
    :return:
    """
    temp1 = pd.Series(np.where((df.returns < 0), u.stddev(df.returns, 20), df.close), index=df.index)
    df['alpha1'] = (u.ts_argmax(temp1 ** 2, 5)) - 0.5
    return df


def alpha2(df):
    """
    Alpha#2
    (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    """

    tmp_1 = u.delta(np.log(df.volume + 1e-6), 2)
    tmp_2 = ((df.close - df.open) / df.open)
    df['alpha2'] = -1 * u.corr(tmp_1, tmp_2, 10)

    return df


def alpha9(df, window=1200):
    """
    Alpha#9
    ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
    ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    """
    tempd1 = df.close.pct_change()
    tempmin = u.ts_min(tempd1, 5)
    tempmax = u.ts_max(tempd1, 5)
    df['alpha9'] = pd.Series(np.where(tempmin > 0, tempd1, np.where(tempmax < 0, tempd1, (-1 * tempd1))), df.index)
    df = compute_zscore(df, 'alpha9', window)
    df = df.drop(columns=['alpha9'])
    df = df.rename(columns={'zscore_alpha9': 'alpha9'})
    return df


def alpha25(df, window=1200):
    """
    Alpha#25
    rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    """
    df['alpha25'] = (((-1 * df.returns) * u.adv(df, 20)) * df.vwap) * (df.high - df.close)
    df = compute_zscore(df, 'alpha25', window)
    df = df.drop(columns=['alpha25'])
    df = df.rename(columns={'zscore_alpha25': 'alpha25'})
    return df


def alpha32(df):
    """
    Alpha#32
    (scale(((sum(close, 7) / 7) - close)) +
    (20 * scale(correlation(vwap, delay(close, 5), 230))))
    """
    temp1 = u.scale(((u.ts_sum(df.close, 7) / 7) - df.close))
    temp2 = (20 * u.scale(u.corr(df.vwap, u.delay(df.close, 5), 230)))
    df['alpha32'] = temp1 + temp2
    return df


def alpha46(df):
    """
    Alpha#46
    ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) :
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
    ((-1 * 1) * (close - delay(close, 1)))))
    """
    decision1 = (0.25 < (
            ((u.delay(df.log_close, 20) - u.delay(df.log_close, 10)) / 10) - (
            (u.delay(df.log_close, 10) - df.log_close) / 10)))
    decision2 = ((((u.delay(df.log_close, 20) - u.delay(df.log_close, 10)) / 10) - (
            (u.delay(df.log_close, 10) - df.log_close) / 10)) < 0)
    iffalse = ((-1 * 1) * (df.log_close - u.delay(df.log_close, 1)))
    df['alpha46'] = pd.Series(np.where(decision1, (-1 * 1), np.where(decision2, 1, iffalse)), index=df.index)
    return df


def alpha51(df):
    """
    Alpha#51
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
    < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    condition = ((((u.delay(df.log_close, 20) - u.delay(df.log_close, 10)) / 10)
                  - ((u.delay(df.log_close, 10) - df.log_close) / 10)) < (-1 * 0.05))
    df['alpha51'] = pd.Series(np.where(condition, 1, ((-1 * 1) * (df.log_close - u.delay(df.log_close, 1)))), df.index)
    return df


def alpha35(df):
    """
    Alpha#35
    ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) *
    (1 - Ts_Rank(returns, 32)))
    """
    df['alpha35'] = ((u.ts_rank(df.volume, 32) * (1 - u.ts_rank(((df.close + df.high) - df.low), 16)))
                     * (1 - u.ts_rank(df.returns, 32)))
    return df


def alpha95(df, window=1200):
    temp1 = (df.open - u.ts_min(df.open, 12))
    df['alpha95'] = temp1
    df = compute_zscore(df, 'alpha95', window)
    df = df.drop(columns=['alpha95'])
    df = df.rename(columns={'zscore_alpha95': 'alpha95'})
    return df


def alpha101(group, fast_period=12, slow_period=26, window=1200):
    """
    计算APO的滚动标准分
    """
    group = group.copy()
    group['APO'] = talib.APO(group['close'], fast_period, slow_period,
                             matype=0)
    group = compute_zscore(group, column='APO', window=window)
    group = group.drop(columns='APO')
    group = group.rename(columns={'zscore_APO': 'alpha101'})
    return group


def alpha102(group: pd.DataFrame, time_period: int = 12) -> pd.DataFrame:
    """
    RSRI指标，对过去n天的最高价最低价去作线性回归
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
    group['alpha102'] = (group['beta'] - beta_mean) / beta_std
    #group['alpha102'] = group['beta']
    return group


def alpha103(group: pd.DataFrame, time_period: int = 12) -> pd.DataFrame:
    """
    计算收益率的偏度和峰度.目前来看只有偏度有用
    """
    group = group.copy()
    # 计算收益率（百分比变化率）
    group['return'] = group['close'].pct_change()
    # 利用滚动窗口计算收益率的偏度和峰度，只有当窗口内数据足够时才计算（否则返回 NaN）
    group['alpha103'] = group['return'].rolling(window=time_period, min_periods=time_period).skew()
    # group['kurt'] = group['return'].rolling(window=time_period, min_periods=time_period).kurt()
    # group['skew_kurt_ratio'] = group['skew']/group['kurt']
    # group['skew_kurt_ratio_std'] = group['skew_kurt_ratio'].rolling(time_period).mean()
    return group


def alpha104(group):
    """
    计算流动性指标amihud
    """
    group = group.copy()
    group['return'] = group['close'].pct_change()
    group['amount'] = group['close'] * group['volume']
    group['ILLIQ'] = np.abs(group['return']) / (group['amount'])
    group['alpha104'] = np.where(group['amount'] == 0, np.nan,
                                 np.abs(group['return']) / group['amount'])
    group = group.drop(columns=['ILLIQ', 'amount'])
    return group


def alpha105(group, time_period=300):
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
    group['returns'] = group['close'].pct_change()  # (Close[t] - Close[t-1]) / Close[t-1]
    group['volume_change'] = group['volume'].pct_change()
    group['alpha105'] = group['returns'].rolling(window=time_period).corr(group['volume_change'])
    group = group.drop(columns='volume_change')
    return group


def alpha106(df, window=30):
    """
    在 df 中计算以下 VWAP 因子:
      1) vwap_roll: 滚动VWAP
      2) vwap_deviation: (close - vwap) / vwap
      3) vwap_slope_diff: 简单差分法计算斜率
      4) vwap_slope_reg: 线性回归法计算斜率

    参数:
    - df: 必须包含 ['close', 'volume'] 列, 索引建议为时间戳
    - window: int, 滚动窗口大小 (单位: 条/分钟/周期数)

    返回: 新的 DataFrame, 多列因子值
    """
    df = df.copy()

    # 1) 计算滚动VWAP
    #   对 (close*volume) 做 rolling(window).sum()，除以 volume的 rolling(window).sum()
    rolling_pv = (df['close'] * df['volume']).rolling(window).sum()
    rolling_v = df['volume'].rolling(window).sum()
    df['vwap_roll'] = rolling_pv / rolling_v  # 滚动VWAP
    df['alpha106'] = (df['close'] - df['vwap_roll']) / df['vwap_roll']
    df = df.drop(columns=['vwap_roll'])
    return df


def calc_ma_factors(df, price_col='close', ma_window=20):
    """
    给定 DataFrame, 计算某个周期 ma_window 的三项因子:
    1) 斜率 Slope
    2) 连续上升天数 ConsecutiveUp
    3) 回踩次数 TouchCount (rolling touch_window)
    返回包含这三个字段的一个子 DataFrame
    """
    # 移动平均
    ma_col = f'MA_{ma_window}'
    df[ma_col] = df[price_col].rolling(ma_window).mean()

    # 斜率
    slope_col = f'Slope_{ma_window}'
    df[slope_col] = (df[ma_col] - df[ma_col].shift(ma_window)) / df[ma_col].shift(ma_window) / ma_window

    # 连续上升天数
    up_flag_col = f'MA_up_{ma_window}'
    df[up_flag_col] = df[ma_col] > df[ma_col].shift(1)
    consecutive_col = f'ConsecutiveUp_{ma_window}'
    cnt = 0
    consecutive_list = []
    for val in df[up_flag_col]:
        if val:
            cnt += 1
        else:
            cnt = 0
        consecutive_list.append(cnt)
    df[consecutive_col] = consecutive_list

    # 返回需要的列
    return df[[ma_col, slope_col, consecutive_col]]


def alpha107(df):
    """
    计算多个均线周期 (5, 10, 15) 的因子，并合成一个综合因子 `Factor_multi`

    参数:
        df (pd.DataFrame): 包含 'asset' 和 'close' 的数据，按资产进行分组计算因子。

    返回:
        pd.DataFrame: 计算后包含新因子的 DataFrame
    """
    ma_list = [5, 10, 15]
    df = df.copy()
    result_list = []
    original_cols = df.columns.tolist()  # 转换为 list
    original_cols.append('Factor_multi')  # 现在可以 append 了

    for asset, df_multi in df.groupby('asset'):
        df_multi = df_multi.copy()
        all_factor_cols = []

        for m in ma_list:
            factor_cols = calc_ma_factors(df_multi, price_col='close', ma_window=m).columns
            slope_col = f'Slope_{m}'
            consec_col = f'ConsecutiveUp_{m}'

            # 计算单周期因子 F_m
            df_multi[f'Factor_{m}'] = 0.8 * df_multi[slope_col] + 0.2 * df_multi[consec_col]
            all_factor_cols.append(f'Factor_{m}')

        # 计算多周期综合因子
        df_multi['Factor_multi'] = df_multi[all_factor_cols].mean(axis=1)
        df_multi = df_multi[original_cols]
        result_list.append(df_multi)

    # 合并所有资产的结果
    df_result = pd.concat(result_list, axis=0)
    df_result = df_result.rename(columns={'Factor_multi':'alpha107'})
    return df_result


def alpha108(df, window=1200, vol_window=30):
    """
    计算高斯波动率（基于滚动标准差）并标准化

    参数：
        df: DataFrame，包含价格数据
        window: 用于计算 Z-score 的标准化窗口
        vol_window: 计算波动率的滚动窗口大小（例如 60 表示用过去 60 期计算波动率）

    返回:
        df: 添加了 'gaussian_volatility' 列的 DataFrame
    """
    df['gk_volatility'] = np.sqrt(
        (0.5 * (np.log(df['high']) - np.log(df['low'])) ** 2) -
        ((2 * np.log(2) - 1) * (np.log(df['close']) - np.log(df['open'])) ** 2)
    )

    # 计算滚动窗口高斯波动率
    df['gaussian_volatility'] = df['gk_volatility'].rolling(vol_window).mean()
    # 计算 Z-score 标准化
    df = compute_zscore(df, 'gaussian_volatility', window)
    # 清理临时列
    df = df.drop(columns=['gaussian_volatility'])
    df = df.rename(columns={'zscore_gaussian_volatility': 'alpha108'})

    return df


def alpha109(df):
    df['alpha124'] = df['medium_price'].diff()
    return df

def alpha110(df):
    for i in range(1,21):
        df[f'alpha110_{i}'] = df['close'].pct_change(i)
    return df
def alpha111(df):
    for i in range(1,21):
        df[f'alpha111_{i}'] = df['high'].pct_change(i)
    return df
def alpha112(df):
    for i in range(1,21):
        df[f'alpha112_{i}'] = df['low'].pct_change(i)
    return df
def alpha113(df):
    for i in range(1,21):
        df[f'alpha113_{i}'] = df['open'].pct_change(i)
    return df
def alpha114(df):
    for i in range(1,21):
        df[f'alpha114_{i}'] = df['vwap'].pct_change(i)
    return df
from scipy.stats import kurtosis, skew


def compute_minute_distribution_metrics(df, period=30, bins=20):
    """
    在 'df' 中对每个长度为 'period' 的滚动窗口计算分钟收益率分布指标。

    :param df: 必须包含 ['close', 'volume'] 列的 DataFrame (分钟级)
               index 建议为时间戳 (DatetimeIndex)，或至少可以切片
    :param period: 滚动窗口大小 (int)，单位：条/根 (例如 30 表示 30 条分钟数据)
    :param bins: 收益率分箱数量，用于计算分布形状(峰度/偏度/标准差)
    :return: 一个 DataFrame，每行对应一个窗口结束时刻(索引)，
             各列包括:
                ['doc_kurt', 'doc_skew', 'doc_std',
                 'doc_vol_pdf60', 'doc_vol_pdf70', 'doc_vol_pdf80',
                 'doc_vol_pdf90', 'doc_vol_pdf95',
                 'doc_vol10_ratio', 'doc_vol5_ratio', 'doc_vol50_ratio']
    """

    # 为安全起见复制一份，避免对原数据修改
    df = df.copy()

    # 内部小函数：对一个子 DataFrame 计算分布指标
    def _calc_distribution_metrics(subdf):
        # 计算分钟收益率
        subdf['returns'] = subdf['close'].pct_change()
        subdf.dropna(subset=['returns'], inplace=True)

        # 如果子集太小，直接返回一堆 NaN
        if len(subdf) < 2:
            return {
                'doc_kurt': np.nan,
                'doc_skew': np.nan,
                'doc_std': np.nan,
                'doc_vol_pdf60': np.nan,
                'doc_vol_pdf70': np.nan,
                'doc_vol_pdf80': np.nan,
                'doc_vol_pdf90': np.nan,
                'doc_vol_pdf95': np.nan,
                'doc_vol10_ratio': np.nan,
                'doc_vol5_ratio': np.nan,
                'doc_vol50_ratio': np.nan,
            }

        # ========== (1) 根据收益率分箱，得到“分布” ==========
        r_min, r_max = subdf['returns'].min(), subdf['returns'].max()
        # 建立分箱区间
        bin_edges = np.linspace(r_min, r_max, bins + 1)

        # 分箱
        subdf['r_bin'] = pd.cut(subdf['returns'], bins=bin_edges, include_lowest=True)
        vol_per_bin = subdf.groupby('r_bin', observed=False)['volume'].sum()

        dist_array = vol_per_bin.values
        # 计算峰度、偏度、标准差
        doc_kurt = kurtosis(dist_array, fisher=True, bias=False)
        doc_skew = skew(dist_array, bias=False)
        doc_std = np.std(dist_array, ddof=1)

        # ========== (2) 收益率分位数 ==========
        doc_vol_pdf60 = subdf['returns'].quantile(0.60)
        doc_vol_pdf70 = subdf['returns'].quantile(0.70)
        doc_vol_pdf80 = subdf['returns'].quantile(0.80)
        doc_vol_pdf90 = subdf['returns'].quantile(0.90)
        doc_vol_pdf95 = subdf['returns'].quantile(0.95)

        # ========== (3) 高收益区间成交量占比 ==========
        # 排序后取最高收益的 N 个 Bar
        subdf_sorted = subdf.sort_values('returns', ascending=False)
        top10_vol = subdf_sorted.head(10)['volume'].sum()
        top5_vol = subdf_sorted.head(5)['volume'].sum()
        top50_vol = subdf_sorted.head(50)['volume'].sum()
        total_vol = subdf['volume'].sum() if subdf['volume'].sum() != 0 else np.nan

        doc_vol10_ratio = top10_vol / total_vol if total_vol > 0 else np.nan
        doc_vol5_ratio = top5_vol / total_vol if total_vol > 0 else np.nan
        doc_vol50_ratio = top50_vol / total_vol if total_vol > 0 else np.nan

        return {
            'doc_kurt': doc_kurt,
            'doc_skew': doc_skew,
            'doc_std': doc_std,
            'doc_vol_pdf60': doc_vol_pdf60,
            'doc_vol_pdf70': doc_vol_pdf70,
            'doc_vol_pdf80': doc_vol_pdf80,
            'doc_vol_pdf90': doc_vol_pdf90,
            'doc_vol_pdf95': doc_vol_pdf95,
            'doc_vol10_ratio': doc_vol10_ratio,
            'doc_vol5_ratio': doc_vol5_ratio,
            'doc_vol50_ratio': doc_vol50_ratio,
        }

    # ========== (4) 在每个 period 滚动窗口上计算 ==========
    results = []
    # 我们在 i 从 [period, len(df)] 范围滚动
    for i in range(period, len(df) + 1):
        subdf = df.iloc[i - period: i]  # 取当前窗口的数据
        metrics_dict = _calc_distribution_metrics(subdf)
        # 用当前窗口的“末行时间”作为这条统计的索引
        end_time = df.index[i - 1]
        print(end_time)
        metrics_dict['end_time'] = end_time
        results.append(metrics_dict)

    # 合并为 DataFrame
    df_rolling = pd.DataFrame(results).set_index('end_time')

    return df_rolling


import statsmodels.api as sm  # 用于线性回归 (可选)
