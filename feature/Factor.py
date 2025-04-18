from multiprocessing import Pool
import talib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils as u
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import inspect
from functools import partial
class FactorConstructor:
    def __init__(self, df):
        self.df = df

    def _compute_alpha_parallel(self, alpha_func, n_jobs=4, **kwargs)->pd.DataFrame:
        grouped = [group for _, group in self.df.groupby('asset')]
        # 包装 alpha_func（固定 kwargs）
        func_with_kwargs = partial(alpha_func, **kwargs)
        with Pool(n_jobs) as pool:
            results = pool.map(func_with_kwargs, grouped)
        final_df = pd.concat(results).sort_index()
        final_df.index = self.df.index
        return final_df

    def run_alphas(self, alpha_names: list, n_jobs=16, **kwargs):
        alpha_funcs = {
            name: getattr(self.__class__, name)
            for name in alpha_names
        }

        for name, func in alpha_funcs.items():
            print(f"Computing {name}...")
            self.df[name] = self._compute_alpha_parallel(func, n_jobs=n_jobs, **kwargs)[name]

    def run_all_alphas(self, n_jobs=16, **kwargs):
        """
        自动运行所有以 alpha 开头的静态方法
        """
        alpha_funcs = [
            (name, method)
            for name, method in inspect.getmembers(self.__class__, predicate=inspect.isfunction)
            if name.startswith("alpha")
        ]

        df_out = self.df.copy()
        for name, func in alpha_funcs:
            print(f"Computing {name}...")
            df_out = self._compute_alpha_parallel(func, n_jobs=n_jobs, **kwargs)
        return df_out

    def get_data(self):
        return self.df

    @staticmethod
    def alpha3(df,window=20):
        """
        乖离率，检验距离均线偏离程度
        """
        df = df.copy()
        df['MA'] = df['close'].rolling(window=window).mean()
        df['alpha3'] = (df['close'] - df['MA']) / df['MA']
        df.drop(columns=['MA'], inplace=True)
        return df  # 如果只需要返回这个因子
    @staticmethod
    def alpha4(df, window=20):
        """
        过去 window 个 bar 中上涨 bar 的占比
        """
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['up'] = (df['return'] > 0).astype(int)
        df['alpha4'] = df['up'].rolling(window).sum() / window
        df.drop(columns=['return', 'up'], inplace=True)
        return df

    @staticmethod
    def alpha5(df, window=20):
        df = df.copy()
        df['alpha5'] = (df['medium_price'] - df['low']) / (df['high'] - df['low'])
        df['alpha5'] = df['alpha5'].rolling(window).mean()
        return df

    @staticmethod
    def alpha6(df, window=20):
        df = df.copy()
        df['alpha6'] = (df['medium_price'] - df['vwap']) / df['vwap']
        df['alpha6'] = df['alpha6'].rolling(window).mean()
        return df

    @staticmethod
    def alpha7(df, fast_period=12, slow_period=26, window=1200):
        """
        计算APO的滚动标准分
        """
        group = df.copy()
        group['APO'] = talib.APO(group['close'], fast_period, slow_period,
                                 matype=0)
        group = compute_zscore(group, column='APO', window=window)
        group = group.drop(columns='APO')
        group = group.rename(columns={'zscore_APO': 'alpha7'})
        return group

    @staticmethod
    def alpha8(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        RSRI指标，对过去n天的最高价最低价去作线性回归
        针对单个资产的数据，使用过去 time_period 天（或 K 线）的最低价和最高价
        进行最小二乘回归，计算回归斜率 beta，
        beta 定义为：beta = Cov(low, high) / Var(low)
        """
        group = df.copy()
        # 计算滚动窗口内 'low' 与 'high' 的协方差和 'low' 的方差
        rolling_cov = group['low'].rolling(window=window, min_periods=window).cov(group['high'])
        rolling_var = group['low'].rolling(window=window, min_periods=window).var()
        # 计算 beta
        beta_raw = np.where(rolling_var == 0, np.nan, rolling_cov / rolling_var)
        beta = pd.Series(beta_raw, index=group.index).ffill()
        group['alpha8'] = beta
        # beta_mean = group['beta'].rolling(window=1200, min_periods=time_period).mean()
        # beta_std = group['beta'].rolling(window=1200, min_periods=time_period).std()
        # group['alpha102'] = (group['beta'] - beta_mean) / beta_std
        return group

    @staticmethod
    def alpha9(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算收益率的偏度和峰度.目前来看只有偏度有用
        换成tick bar后 不知道为什么似乎峰度效果好
        """
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # df['alpha9'] = df['log_ret'].rolling(window).apply(lambda x: skew(x, bias=False), raw=True)
        df['alpha9'] = df['log_ret'].rolling(window).apply(lambda x: kurtosis(x, fisher=True, bias=False), raw=True)
        df.drop(columns=['log_ret'], inplace=True)
        return df

    @staticmethod
    def alpha10(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算流动性指标amihud
        本质为过去一根k线单位dollar推动的涨幅
        """
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['amount'] = df['close'] * df['volume']
        df['ILLIQ'] = np.abs(df['return']) / (df['amount'])
        df['alpha10'] = np.where(df['amount'] == 0, np.nan,
                                     np.abs(df['return']) / df['amount'])
        df = df.drop(columns=['ILLIQ', 'amount'])
        return df

    @staticmethod
    def alpha11(df: pd.DataFrame, time_period=300):
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
        # 目前来看在tick数据上表现不太好
        # 复制数据，防止对原数据修改
        group = df.copy()
        group['returns'] = group['close'].pct_change()
        group['dollar_volume'] = group['volume'].pct_change()
        group['alpha11'] = group['returns'].rolling(window=time_period).corr(group['dollar_volume'])
        group.drop(columns=['returns', 'dollar_volume'], inplace=True)
        return group

    @staticmethod
    def alpha12(df, window=30):
        """
        计算vwap的滚动
        """
        df['alpha12'] = (df['vwap'] - df['medium_price']) / df['vwap']
        df['alpha12'] = df['alpha12'].rolling(window=window).mean()
        return df
    @staticmethod
    def alpha13(df, window=1200, vol_window=30):
        """
        计算高斯波动率（基于滚动标准差）并标准化

        参数：
            df: DataFrame，包含价格数据
            window: 用于计算 Z-score 的标准化窗口
            vol_window: 计算波动率的滚动窗口大小（例如 60 表示用过去 60 期计算波动率）

        返回:
            df: 添加了 'gaussian_volatility' 列的 DataFrame
        """
        df = df.copy()
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
        df = df.rename(columns={'zscore_gaussian_volatility': 'alpha13'})

        return df

    @staticmethod
    def alpha14(df, window=20):
        df = df.copy()
        total_volume = df['buy_volume'] + df['sell_volume']
        imbalance = (df['buy_volume'] - df['sell_volume']) / total_volume.replace(0, np.nan)
        df['alpha14'] = imbalance.rolling(window).mean()
        return df
    @staticmethod
    def alpha15(df, window=20):
        df = df.copy()
        df['alpha15'] = df['tick_interval_mean'].ewm(span=window).mean()
        return df
    @staticmethod
    def alpha16(df, window=20):
        df_copy = df.copy()
        cols = []
        for i in range(1, window + 1):
            colname = f'alpha111_{i}'
            cols.append(colname)
            df_copy[colname] = df['close'].pct_change(i)
        df['alpha16'] = apply_pca(df_copy, cols)
        return df

    @staticmethod
    def alpha17(df, window=20):
        df_copy = df.copy()
        cols = []
        for i in range(1, window + 1):
            colname = f'alpha112_{i}'
            cols.append(colname)
            df_copy[colname] = df['high'].pct_change(i)
        df['alpha17'] = apply_pca(df_copy, cols)
        return df

    @staticmethod
    def alpha18(df, window=20):
        df_copy = df.copy()
        cols = []
        for i in range(1, window + 1):
            colname = f'alpha113_{i}'
            cols.append(colname)
            df_copy[colname] = df['open'].pct_change(i)
        df['alpha18'] = apply_pca(df_copy, cols)
        return df

    @staticmethod
    def alpha19(df, window=20):
        df_copy = df.copy()
        cols = []
        for i in range(1, window + 1):
            colname = f'alpha114_{i}'
            cols.append(colname)
            df_copy[colname] = df['low'].pct_change(i)
        df['alpha19'] = apply_pca(df_copy, cols)
        return df

    @staticmethod
    def alpha20(df, window=20):
        df_copy = df.copy()
        cols = []
        for i in range(1, window + 1):
            colname = f'alpha115_{i}'
            cols.append(colname)
            df_copy[colname] = df['vwap'].pct_change(i)
        df['alpha20'] = apply_pca(df_copy, cols)
        return df

    @staticmethod
    def alpha21(df, window=20):
        df_copy = df.copy()
        cols = []
        for i in range(1, window + 1):
            colname = f'alpha116_{i}'
            cols.append(colname)
            df_copy[colname] = df['volume'].pct_change(i)
        df['alpha21'] = apply_pca(df_copy, cols)
        return df

    @staticmethod
    def alpha22(df, window=20):
        df['alpha22'] = df['reversals']
        return df
    @staticmethod
    def alpha23(df, window=20):
        df['alpha23'] = df['skewness']
        return df
    @staticmethod
    def alpha24(df, window=20):
        df['alpha24'] = df['kurtosis']
        return df
    @staticmethod
    def alpha25(df, window=20):
        df['alpha25'] = df['up_move_ratio']
        return df


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


def apply_pca(df, cols, n_components=3):
    group = df.copy()
    X = group[cols].dropna()
    if len(X) < 10:
        return np.full(len(group), np.nan)

    split = len(X) // 2
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_pca_full = np.vstack([X_train_pca, X_test_pca])
    result = np.full(len(group), np.nan)
    result[-len(X_pca_full):] = X_pca_full[:, 0]  # 提取第一主成分
    return result