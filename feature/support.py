from multiprocessing import Process, Pool
import pandas as pd
from numba import njit
import numpy as np
import plotly.graph_objects as go
from IC_calculator import compute_zscore, compute_ic
from feature.read_large_files import select_assets, map_and_load_pkl_files


def rolling_autocorr_factor(df, window=20):
    """
    在一个滚动窗口里计算一阶自相关系数，并把它作为因子/特征。
    假设 df 有一个 'return' 列代表收益率 (r_t)。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列 'return'，index 是时间。
    window : int
        滚动窗口长度（单位：期数，比如日频就 20 天）。

    Returns
    -------
    pd.DataFrame
        多一列 'autocorr_factor' 表示滚动自相关因子。
    """
    df = df.copy()

    # 简单方法：用 rolling.apply + 自定义函数来算 autocorr
    def autocorr_func(x):
        return x.autocorr(lag=1)  # pandas.Series 自带autocorr方法

    df['autocorr_factor'] = (
        df['close']
        .rolling(window=window, min_periods=window)
        .apply(autocorr_func, raw=False)
    )

    return df


def compute_intermediate_momentum_monthly(
        df: pd.DataFrame,
        start_month: int = 1,
        end_month: int = 1
) -> pd.DataFrame:
    """
    在 15 分钟级别的数据上，先计算“月度回报率”，再根据过去 start_month ~ end_month 的月度回报率
    累乘得到一个动量指标。

    Parameters
    ----------
    df : pd.DataFrame
        如果是 MultiIndex: [time, asset]，则必须至少包含 'close' 列。
        如果还不是 MultiIndex，请先:
            df = df.set_index(['time','asset']).sort_index()
    start_month : int
        动量计算的起始滞后月 (例如 7)。
    end_month : int
        动量计算的结束滞后月 (例如 12)。

    Returns
    -------
    pd.DataFrame
        在原 DataFrame 基础上新增两列:
          - 'monthly_return'：表示“从 t-1个月 到 t 这一整月”的回报率
          - 'momentum_n_m'：表示过去 n 到 m 个月 (start_month ~ end_month) 的复合收益率
    """

    # 确保索引和排序

    # 1) 先计算单月回报率：月度间隔近似为 2880 个 15 分钟 bar
    intervals_per_month = 30  # 30天 × 24小时 × (60/15)=4
    # groupby('asset') 是为了每个币种各自 shift，而不互相干扰
    df['monthly_return'] = df['close'] / df['close'].shift(intervals_per_month) - 1

    # 2) 对于 “过去 n 到 m 个月”的复合收益率，需要依次抓取第 t-7、t-8... t-12 个月的 monthly_return
    #    然后做累乘
    #    注意：这里“t-7 个月的回报”就是 row t-7×intervals_per_month 对应的 monthly_return
    #    我们可以对 monthly_return 再 shift(k×intervals_per_month) 或者直接把 row t 的 monthly_return
    #    看成 "从 t-1个月 到 t" 的回报率。下面使用 shift 让“过去k个月的月度回报”对齐到当前行

    for k in range(start_month, end_month + 1):
        shift_k = k  # 月份距离
        # 将“t-k 个月”的 monthly_return 移到当前行
        df[f'l{k}'] = df['monthly_return'].shift(shift_k)

    # 3) 累乘得到复合收益率
    df['momentum_n_m'] = 1.0
    for k in range(start_month, end_month + 1):
        df['momentum_n_m'] *= (1 + df[f'l{k}'])
    df['momentum_n_m'] -= 1

    return df


def process_wrapper(data, func, n_jobs=16):
    grouped = []
    for asset, df_asset in data.groupby('asset'):
        grouped.append(df_asset)
    with Pool(n_jobs) as pool:
        results = pool.map(func, grouped)
    df_out = pd.concat(results, axis=0)
    df_out.sort_index(inplace=True)
    return df_out


@njit
def compute_ha_open(ha_close, original_open):
    ha_open = np.empty_like(ha_close)
    ha_open[0] = original_open[0]  # 第一根K线的HA_Open
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    return ha_open


def heikin_ashi_transform(df):
    """
    向量化计算 Heikin Ashi 变换
    输入:
        df: DataFrame，包含 ['time', 'open', 'high', 'low', 'close']
    输出:
        DataFrame，新增 ['ha_open', 'ha_high', 'ha_low', 'ha_close']
    """
    ha_close = (df['open'].values + df['high'].values + df['low'].values + df['close'].values) / 4

    # 计算 HA_Open (需要递归计算, 用 numba 加速)
    ha_open = compute_ha_open(ha_close, df['open'].values)

    # 计算 HA_High 和 HA_Low (向量化计算)
    ha_high = np.maximum(df['high'].values, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(df['low'].values, np.minimum(ha_open, ha_close))

    # 生成新的 DataFrame
    df['open'] = ha_open
    df['high'] = ha_high
    df['low'] = ha_low
    df['close'] = ha_close

    return df


def ma_smooth(df):
    columns = ['open', 'high', 'low', 'close']
    df_smooth = df.copy()  # 避免修改原始 df
    for column in columns:
        df_smooth[column] = df_smooth[column].ewm(span=10).mean()
    return df_smooth


def cusum_filter(df, threshold=0.03, drift=0):
    s_pos, s_neg = 0, 0
    series = df['close'].pct_change().values
    signals = []
    for i, r in enumerate(series):
        s_pos = max(0, s_pos + r - drift)
        s_neg = min(0, s_neg + r + drift)
        if s_pos > threshold:
            signals.append(1)  # 正向趋势拐点
            s_pos, s_neg = 0, 0
        elif s_neg < -threshold:
            signals.append(-1)  # 负向趋势拐点
            s_pos, s_neg = 0, 0
        else:
            signals.append(0)

    df['signal'] = signals
    return df


if __name__ == "__main__":
    start = "2020-1-1"
    end = "2023-12-31"
    assets = select_assets(start_time=start, spot=True, n=50)
    print(assets)

    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    data['future_return'] = data.groupby('asset')['close'].apply(
        lambda x: x.shift(-10) / x - 1).droplevel(0)
    # data['future_return'] = data.groupby(level='time')['future_return'].rank(
    #     method='dense',  # 或者 'first','average','min','max' 等
    #     ascending=False  # 如果你想把收益率高的排在前面，就用 False
    # )
    print(data)
    data = process_wrapper(data, func=rolling_autocorr_factor)
    # data = process_wrapper(data, func=cusum_filter)
    print(data)
    # columns = ['momentum']
    # for column in data.columns:
    #     if column not in ['open', 'close', 'high', 'low', 'APO', 'RSI', 'returns', 'volume', 'return', 'log_close',
    #                       'future_return', 'vwap', 'amount', 'beta', 'downsidevolatility', 'upsidevolatility',
    #                       'volumestd']:
    #         ic = compute_ic(df=data, feature_column=column, return_column='future_return',method='pearson')
    #         # ic = compute_ic(df=data, feature_column='zscore_RSI', return_column='future_return')
    #         print(column, "IC_MEAN:", np.mean(ic), "IR", np.mean(ic) / np.std(ic))
    # # for asset, df in data.groupby('asset'):
    #     import plotly.graph_objects as go
    #
    #     # 绘制K线图
    #     fig = go.Figure(data=[
    #         go.Candlestick(
    #             x=df.index.get_level_values('time'),
    #             open=df['open'],
    #             high=df['high'],
    #             low=df['low'],
    #             close=df['close'],
    #             name='Price'
    #         )
    #     ])
    #
    #     # 标记买入信号 (signal == 1)
    #     buy_signals = df[df['signal'] == 1]
    #     fig.add_trace(
    #         go.Scatter(
    #             x=buy_signals.index.get_level_values('time'),
    #             y=buy_signals['low'] * 0.995,  # 在最低点稍下方标记
    #             mode='markers',
    #             marker=dict(color='green', size=10, symbol='triangle-up'),
    #             name='Buy Signal'
    #         )
    #     )
    #
    #     # 标记卖出信号 (signal == -1)
    #     sell_signals = df[df['signal'] == -1]
    #     fig.add_trace(
    #         go.Scatter(
    #             x=sell_signals.index.get_level_values('time'),
    #             y=sell_signals['high'] * 1.005,  # 在最高点稍上方标记
    #             mode='markers',
    #             marker=dict(color='red', size=10, symbol='triangle-down'),
    #             name='Sell Signal'
    #         )
    #     )
    #
    #     fig.update_layout(
    #         xaxis_rangeslider_visible=False,
    #         title='Candlestick Chart with Signals'
    #     )
    #
    #     fig.show()
    #
    #     break
