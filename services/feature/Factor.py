import math
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
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
    def alpha1(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        meam reversion都有一个问题
        当出现大量信号的时候，会几乎全仓
        这个时候市场突破导致大仓位巨大亏损
        而平时只能小幅度稀疏仓位上行
        区间：
        long_range = (-20,-4)
        short_range = (4,20)
        """
        df = df.copy()
        # df['return'] = df['close'].pct_change(window)
        # returns = np.sign(df['return'])*np.log1p(abs(df['return']))**2
        up_ratio = df['up_move_ratio'].rolling(window=window).sum()
        df['alpha1'] =up_ratio
        return df

    @staticmethod
    def alpha2(df: pd.DataFrame,window = 60,rolling_window = 1200) -> pd.DataFrame:
        """
        计算流动性指标amihud
        本质为过去n根k线单位dollar推动的涨幅
        短时间内 如果很少金钱就推动了很大涨幅 那么容易反转
        默认参数
        long_range = (-1,-0.3)
        short_range = (0.3,1)
        """
        df = df.copy()
        ret = df['close'].pct_change(window)
        df['return'] = ret
        df['dollar'] = df['volume'] * df['vwap']
        df['amount'] = np.log10(df['dollar'].rolling(window=window).sum())
        df['factor'] = np.where(
            df['amount'] == 0,
            np.nan,
            df['return'] *1e3/ df['amount']
        )
        factor =df['factor']
        # df['std'] = df['close'].rolling(window=window).std() / df['close'].rolling(window=window * 10).std()
        df['alpha2'] = -factor
        return df

    @staticmethod
    def alpha3(df,window=60):
        """
        乖离率，检验距离均线偏离程度
        优化下乘以一个主动买入系数
        mean reversion策略需要强风控来控制回撤 不然收益率很难看(卡玛约在1)
        """
        df = df.copy()
        df['MA'] = df['close'].rolling(window=window).mean()
        df['std'] = df['close'].rolling(window=window).std()/df['close'].rolling(window=window*5).std()
        volume = df['volume'].rolling(window=window).mean()
        df['alpha3'] = (-(df['close'] - df['MA']) / df['MA']) * df['std']#过滤低波噪声
        # df['alpha3'] = (-(df['close'] - df['MA']) / df['MA'])*volume_ratio
        df.drop(columns=['MA'], inplace=True)
        return df  # 如果只需要返回这个因子

    @staticmethod
    def alpha4(df: pd.DataFrame, window=120) -> pd.DataFrame:
        """
        主动买入的推动涨幅，
        只有大幅主动买入，且推动涨幅，才认为上涨强烈
        如果大幅卖出，且大幅下跌，则卖出强烈
        long_range = (-10,-5)
        short_range = (5,10)
        注意这类策略需要把持有窗口期和atr稍微开大一些
        """
        df['return'] = (df['close'] - df['close'].shift(window)) / df['close']
        buyer = df['buy_volume']
        seller = df['volume'] - df['buy_volume']
        df['imbalance'] = (buyer - seller)/df['volume']
        df['alpha4'] = 100*df['imbalance'].rolling(window).sum() * np.sign(df['return'])*np.log1p(abs(df['return']))**4

        return df

    @staticmethod
    def alpha5(df, window=60):
        """
        bar中中位成交价格占bar柱的情况
        理论上来说
        """
        df = df.copy()
        df['alpha5'] = (df['medium_price'] - df['low']) / (df['high'] - df['low'])
        df['alpha5'] = -df['alpha5'].rolling(window).mean()
        return df

    @staticmethod
    def alpha6(df, window=60):
        """
        中位成交价和vwap的乖离率
        """
        df = df.copy()
        df['alpha6'] = (df['medium_price'] - df['vwap']) / df['vwap']
        df['alpha6'] = df['alpha6'].rolling(window).mean()
        return df

    @staticmethod
    def alpha7(df, fast_period=30, slow_period=60, window=1200):
        """
        计算APO的滚动标准分
        """
        group = df.copy()
        group['APO'] = talib.APO(group['close'], fast_period, slow_period,
                                 matype=0)
        group = compute_zscore(group, column='APO', window=window)
        group = group.drop(columns='APO')
        group = group.rename(columns={'zscore_APO': 'alpha7'})
        group['alpha7'] = -group['alpha7']
        return group

    @staticmethod
    def alpha8(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
        group = df.copy()
        group['hl_range'] = group['high'] - group['low']
        group['mid_price'] = (group['high'] + group['low']) / 2
        group['log_range'] = np.log(group['high'] / group['low'])
        cov = group['log_range'].rolling(window).cov(group['mid_price'])
        var = group['mid_price'].rolling(window).var()
        beta = np.where(var == 0, np.nan, cov / var)
        group['alpha8'] = beta

        return group

    @staticmethod
    def alpha9(df,window=20):
        df['ret'] = df['close'].pct_change()
        df['reverse_signal'] = -df['ret'].rolling(20).mean()
        df['alpha9'] = (df['reverse_signal'] - df['reverse_signal'].rolling(200).mean()) / df[
            'reverse_signal'].rolling(200).std()
        return df

    @staticmethod
    def alpha10(df, window=60):
        """
        买入强度因子
        好像没啥卵用
        """
        reversed = df['reversals'] / df['tick_nums']
        ticks = df['tick_nums']
        total_reverse = reversed.rolling(window).sum()#买入卖出转变占比
        ret = df['close'].diff(window)
        alpha = -(total_reverse/total_reverse.rolling(window).mean()-1)*np.sign(ret)
        df['alpha10'] = alpha
        return df

    @staticmethod
    def alpha11(df: pd.DataFrame, window=20):
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
        df['alpha11'] = -talib.ROC(df['close'], timeperiod=window)
        return df

    @staticmethod
    def alpha12(df, window=30):
        """
        计算vwap的滚动
        """

        # gain = df['close']
        # df['alpha12'] = (df['vwap'].rolling(window).mean()-df['close'].rolling(window).mean()) / df['close'] * 10
        vals = (df['vwap']- df['close']) / df['close']
        df['alpha12'] = vals.rolling(window).mean()
        return df
    @staticmethod
    def alpha13(df, window=30, vol_window=30):
        """
        """
        # alpha = df['up_move_ratio'].rolling(window).mean()
        # buyer = df['buy_volume'].rolling(window).mean()
        # seller = df['sell_volume'].rolling(window).mean()
        # interval = df['tick_interval_mean'].diff(window) / df['tick_interval_mean'].shift(window)
        ret = df['close'].pct_change(window)
        # imbalance = -(buyer - seller) / df['volume'].shift(window) * interval
        # df['alpha13'] = imbalance
        df['alpha13'] = -ret / (df['buy_volume'].rolling(window).mean() / df['volume'].rolling(window).mean()) # 主动抗住下跌
        return df

    @staticmethod
    def alpha14(df, window=20):
        df = df.copy()
        tick_change = -df['tick_interval_mean'] /df['tick_interval_mean'].rolling(window).mean()
        buyer = (df['buy_volume'].rolling(window).mean() / df['volume'].rolling(window).mean())-0.5
        df['alpha14'] = buyer * tick_change
        return df

    @staticmethod
    def alpha15(df: pd.DataFrame, n=120) -> pd.DataFrame:
        """
        alpha15：过去 n 根 K 线中当前价格高于多少比例的 open（胜率）× 平均盈亏（当前价 - 均值开盘价）
        """
        open_array = df['vwap'].to_numpy()
        close_array = df['close'].to_numpy()
        # 滚动窗口 open
        open_rolling = np.lib.stride_tricks.sliding_window_view(open_array, window_shape=n)
        current_close = close_array[n - 1:]
        # 胜率计算
        win_matrix = current_close[:, None] > open_rolling
        win_ratio = win_matrix.sum(axis=1) / n
        # 平均盈亏计算
        avg_open = open_rolling.mean(axis=1)
        avg_pnl = (current_close - avg_open)/avg_open
        # 相乘得期望因子值
        alpha_val = win_ratio * avg_pnl
        # 填充为与原 df 一致长度
        alpha_col = np.full_like(close_array, np.nan, dtype=np.float64)
        alpha_col[n - 1:] = alpha_val
        df["alpha15"] = -alpha_col
        return df

    @staticmethod
    def alpha16(df, window = 20):
        devitiation = (df['vwap'] - df['medium_price'])/df['close']
        df['alpha16'] = -devitiation.rolling(window=window).mean()
        return df

    @staticmethod
    def alpha17(df, window=120):
        returns = df['close'].pct_change()
        squared_up = returns.where(returns > 0, 0) ** 2
        std_up = squared_up.rolling(window=window).sum().apply(np.sqrt)
        squared_down = returns.where(returns < 0, 0) ** 2
        std_down = squared_down.rolling(window=window).sum().apply(np.sqrt)
        df['alpha17'] = -(std_up-std_down)/(std_up+std_down)

        return df

    @staticmethod
    def alpha18(df, window=120):
        """
          计算趋势强度指标（Trend Strength）
          """
        price = df['close']
        net_change = price.diff(window*100)
        total_movement = price.diff().abs().rolling(window=window*100).sum()
        trend_strength = net_change / total_movement
        volume_score = df['volume'].rolling(window=window).mean()/df['volume']
        df['alpha18'] = trend_strength * volume_score
        return df

    @staticmethod
    def alpha19(
        df: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 120,
        impact_col: str = "max_trade_impact",
        dir_col: str = "max_trade_direction",
        vol_ratio_col: str = "max_trade_vol_ratio",
        out_col: str = "alpha19",
    ) -> pd.DataFrame:
        """
        α19: 过去 n 根 K 线中 **大单推动涨幅**（近端 - 远端）

        定义：
            signed_impact = max_trade_impact * ( 1 if max_trade_direction==0 else -1 )
            bar_score     = signed_impact * max_trade_vol_ratio

            alpha19_t = Σ_{t-short+1}^{t} bar_score  -  Σ_{t-long+1}^{t} bar_score

        参数
        ----
        df : DataFrame
            已经包含 impact, direction, vol_ratio 三列；按时间升序。
        short_window : int
            近端窗口长度（默认 20 根 bar）
        long_window : int
            远端窗口长度（默认 120 根 bar；需 > short_window）
        impact_col, dir_col, vol_ratio_col : str
            对应列名；如你改过名字可在调用时指定。
        out_col : str
            输出列名（默认 'alpha19'）

        返回
        ----
        DataFrame
            原 df 的复制，增加一列 `out_col`
        """
        sign = np.where(df[dir_col] == 0, 1.0, -1.0)
        impact = np.where(df[impact_col]!=np.nan,df[impact_col],0)
        signed_imp =  impact * sign
        bar_score = signed_imp * df[vol_ratio_col].to_numpy()
        df["_bar_score"] = bar_score

        # -------- 2. 近期 / 远期 均值 --------
        recent_sum = (
            df["_bar_score"]
            .rolling(window=short_window, min_periods=short_window)
            .sum()
        )

        cum_long_sum = (
            df["_bar_score"]
            .rolling(window=long_window, min_periods=long_window)
            .sum()
        )

        past_sum = cum_long_sum - recent_sum
        past_len = long_window - short_window

        recent_mean = recent_sum / short_window
        past_mean = past_sum / (past_len + 1e-12)  # 防 0

        # -------- 3. alpha19 --------
        df[out_col] = (recent_mean -recent_mean.rolling(window=600).mean()) / recent_mean.rolling(window=600).std()

        return df.drop(columns="_bar_score")

    @staticmethod
    def alpha20(df, window=200):
        df = df.copy()
        close = df['close']
        net_return = close - close.shift(window)
        path_length = close.diff().abs().rolling(window=window).sum()
        tqf = (net_return.abs() / path_length).replace([np.inf, -np.inf], np.nan)
        df['alpha20'] = tqf
        return df

    @staticmethod
    def alpha21(df, window=200):
        df = df.copy()
        r2_list = [np.nan] * window
        for i in range(window, len(df)):
            y = df['close'].iloc[i - window:i].values.reshape(-1, 1)
            x = np.arange(window).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            r2 = model.score(x, y)
            r2_list.append(r2)
        df['alpha21'] = r2_list
        return df

    @staticmethod
    def alpha22(df, window=120):
        df = df.copy()
        # 回归斜率
        x = np.arange(window)
        slopes = []
        for i in range(window, len(df)):
            y = df['close'].iloc[i - window:i].values
            coef = np.polyfit(x, y, 1)[0]
            slopes.append(coef)
        slopes = [np.nan] * window + slopes
        df['slope'] = slopes

        # 波动率（标准差）
        df['vol'] = df['close'].rolling(window).std()

        df['alpha22'] = -df['slope'] / df['vol']
        return df
    @staticmethod
    def alpha23(df, window=300):
        df = df.copy()
        # 计算滚动最高价
        rolling_max = df['close'].rolling(window=window, min_periods=1).max()
        # 计算每一日的回撤百分比
        drawdown_pct = 100 * (df['close'] - rolling_max) / rolling_max
        # 计算 Ulcer Index：滚动窗口内回撤平方的均值的平方根
        ulcer_index = drawdown_pct.pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)
        # 保存结果
        alpha = np.log1p(ulcer_index)
        df['alpha23'] = alpha
        return df
    @staticmethod
    def alpha24(df, window=20):
        df['alpha24'] = df['kurtosis']
        return df
    @staticmethod
    def alpha25(df, window=20):
        df['alpha25'] = df['up_move_ratio']
        return df
    @staticmethod
    def alpha26(df, window=200):
        """
        检测看看是不是有机构短时间大幅扫货
        终于找到一个是趋势跟随的了
        参数1:
        long_range = (1,5),short_range = (-5,-1),window = 100
        参数2同上，window改为200
        """
        df['return'] = (df['close'] - df['close'].shift(window)) / df['close']
        tick_inv_norm = 1 / df['tick_interval_mean']
        tick_inv_norm = (tick_inv_norm - tick_inv_norm.rolling(window).mean()) / tick_inv_norm.rolling(window).std()
        df['alpha26'] = df['return'] * tick_inv_norm*10

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