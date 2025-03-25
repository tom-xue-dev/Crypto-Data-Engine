import pickle

import numpy as np
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
import matplotlib.pyplot as plt
import pandas as pd
from feature_generation import alpha40 as alpha
import utils as u
from support import process_wrapper
from statsmodels.tsa.stattools import adfuller

def hurst_exponent_rs(series, min_window=10, max_window=None, step=5):
    """
    计算单个时间序列的 Hurst 指数 (R/S分析).

    :param series: 一维时间序列 (list 或 numpy array 或 pandas Series)
    :param min_window: 最小分段长度
    :param max_window: 最大分段长度 (若为 None，则默认为 len(series)//2)
    :param step: 窗口大小的步进
    :return: hurst 指数估计值
    """
    data = np.asarray(series, dtype=float)
    n = len(data)
    if n < 2 * min_window:
        # 序列太短，无法做有效估计
        return np.nan

    if max_window is None:
        max_window = n // 2

    window_sizes = range(min_window, max_window, step)
    rs_vals = []

    for w in window_sizes:
        # 计算在当前 w 下可以分成多少个完整段
        num_segments = n // w
        if num_segments <= 0:
            continue

        r_s_sum = 0.0
        # 对每个 segment 分别计算 R/S
        for i in range(num_segments):
            segment = data[i * w: (i + 1) * w]
            mean_seg = segment.mean()
            # 去均值
            y = segment - mean_seg
            # 累积和
            z = np.cumsum(y)
            # 极差
            R = z.max() - z.min()
            # 标准差 (可用 np.std(segment, ddof=1) or np.sqrt(np.mean(y**2)))
            S = np.std(segment, ddof=1)
            if S != 0:
                r_s_sum += R / S

        # 计算当前窗口下的平均 R/S
        r_s = r_s_sum / num_segments
        rs_vals.append((w, r_s))

    if len(rs_vals) < 2:
        return np.nan

    # 在 log-log 空间线性回归, slope 即为 H
    w_arr = np.log([x[0] for x in rs_vals])
    rs_arr = np.log([x[1] for x in rs_vals])
    slope, intercept = np.polyfit(w_arr, rs_arr, 1)

    return slope  # 即 H


class FactorEvaluator:
    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data : pd.DataFrame
            具有 MultiIndex 的 DataFrame，通常索引形如 (date, asset)，
            并包含因子值和未来收益率两列。
        factor_col : str, default 'factor'
            因子列名。
        ret_col : str, default 'future_return'
            未来收益率列名。
        """
        self.data = data
        self.ic = None  # 存储计算得到的 IC 序列

    def plot_factor_distribution(self, factor_column, n_bins=100, **kwargs):
        """
        绘制self.data中指定因子列（factor_column）的数值分布直方图。
        假设self.data的索引第一层为time，第二层为asset。

        改进内容：
        1. 过滤掉无穷大（正无穷和负无穷）以及缺失值。
        2. 使用Freedman-Diaconis规则动态计算直方图的bins数量。
        """
        # 提取因子数据，并剔除缺失值
        values = self.data[factor_column].dropna()
        # 过滤掉无穷大值
        if 'upper_bound' in kwargs:
            values = np.where(values <= kwargs['upper_bound'], values, kwargs['upper_bound'])
        if 'lower_bound' in kwargs:
            values = np.where(values >= kwargs['lower_bound'], values, kwargs['lower_bound'])

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=n_bins, alpha=0.7, edgecolor='black')
        plt.xlabel(factor_column)
        plt.ylabel("Frequency")
        plt.title(f"{factor_column}'s distribution")
        plt.grid(True)
        plt.show()

    def calculate_cross_ic(self, factor_column, return_column, method='spearman'):
        """
        计算每个时间点上因子值与未来收益的相关性（IC）。
        注意这里是截面IC， 需要保证不同的时间点有多个资产

        参数:
          factor_column: 因子列名称
          return_column: 未来收益列名称
          method: 使用相关性计算的方法，默认使用 'spearman'（也可以选择 'pearson'）

        返回:
          一个 Pandas Series，索引为时间，值为对应时间点的IC。
        """
        ic_list = []
        # 获取所有的时间戳，假设时间在索引的第一层
        time_index = self.data.index.get_level_values(0).unique().sort_values()

        for t in time_index:
            # 获取某个时间点的所有资产数据
            subdata = self.data.loc[t]
            # 计算相关性，注意这里需要确保数据没有缺失值
            if method == 'spearman':
                ic = subdata[factor_column].corr(subdata[return_column], method='spearman')
            else:
                ic = subdata[factor_column].corr(subdata[return_column], method='pearson')
            ic_list.append(ic)
        # 将结果转换成Series，索引为时间
        ic_series = pd.Series(ic_list, index=time_index)
        return ic_series

    def plot_ic(self, factor_column, return_column, method='spearman', **kwargs):
        """
        计算并绘制因子信息系数（IC）时间序列图。

        参数:
          factor_column: 因子列名称
          return_column: 未来收益列名称
          method: 相关性计算方法，默认 'spearman'
          kwargs: 可选的绘图参数，如 figsize, xlabel, ylabel, title 等。
        """
        # 计算IC时间序列
        ic_series = self.calculate_cross_ic(factor_column, return_column, method=method)[100:]
        ic_series = ic_series.expanding(min_periods=1).mean()

        # 解析额外的绘图参数
        figsize = kwargs.get('figsize', (10, 6))
        xlabel = kwargs.get('xlabel', 'Time')
        ylabel = kwargs.get('ylabel', 'IC')
        title = kwargs.get('title', f"{factor_column} Information Coefficient Over Time")

        plt.figure(figsize=figsize)
        plt.plot(ic_series.values, marker='o', linestyle='-')
        # plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_cumulative_ic(self, factor_column, return_column, method='spearman', **kwargs):
        ic_series = self.calculate_cross_ic(factor_column, return_column, method=method).cumsum()
        figsize = kwargs.get('figsize', (10, 6))
        xlabel = kwargs.get('xlabel', 'Time')
        ylabel = kwargs.get('ylabel', 'IC')
        title = kwargs.get('title', f"{factor_column} Cumulative Information Coefficient Over Time")

        plt.figure(figsize=figsize)
        plt.plot(ic_series.values, marker='o', linestyle='-')
        # plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def adf_test(self, column):
        for asset,group in self.data.groupby('asset'):
            result = adfuller(group[column])
            print('ADF Statistic: {:.4f}'.format(result[0]))
            print('p-value: {:.4f}'.format(result[1]))
            print('Number of Lags Used: {}'.format(result[2]))
            print('Number of Observations Used: {}'.format(result[3]))

    def hurst_exponent_for_multiindex(self, target_column, min_window=10, max_window=None, step=5):
        """
        针对多重索引 (time, asset) 的 DataFrame，分别计算每个 asset 在 target_column 上的 Hurst 指数。

        :param df: 带有多重索引 (time, asset) 的 DataFrame
        :param target_column: 要测量 Hurst 指数的列名
        :param min_window: 传递给 hurst_exponent_rs 的最小窗口参数
        :param max_window: 传递给 hurst_exponent_rs 的最大窗口参数
        :param step: 传递给 hurst_exponent_rs 的窗口步长
        :return: 返回一个 pd.Series，index为资产名，values为该资产在 target_column 上的 Hurst 指数
        """
        # 首先确保按 (time, asset) 排序
        df = self.data.sort_index(level=['time', 'asset'])

        results = {}
        # 按 asset 分组
        grouped = df.groupby(level='asset')

        for asset, subdf in grouped:
            # subdf 只有该 asset 的行，索引第一层是 time
            # 取 target_column 这一列作为时间序列
            series = subdf[target_column].values  # 或者 subdf[target_column].to_numpy()

            # 调用单资产 Hurst 计算函数
            H = hurst_exponent_rs(
                series,
                min_window=min_window,
                max_window=max_window,
                step=step
            )

            results[asset] = H

        # 以 Series 形式返回
        return pd.Series(results, name=f'Hurst_{target_column}')
    def plot_factor_stratification(self, quantiles: int = 5):
        """
        根据因子值分层，绘制各层的累计未来收益率表现。

        Parameters
        ----------
        quantiles : int, default 5
            按分位数分层的层数（例如 5 表示分为 5 层，通常称为五分位）。

        说明：
        - 对于每个日期，将该日的所有资产按照因子值进行分层，使用 pd.qcut 分层，
          然后计算各层的平均未来收益率。
        - 计算每日收益率的累计收益率（假设每个收益率代表一个固定时间间隔内的收益），
          即累计收益率 = (1 + r1) * (1 + r2) * ... - 1。
        - 最后将每个分层的累计收益率时间序列绘制出来。
        """
        df = self.data.copy()

        # 对每个时间点内的数据按因子值分层（分位数），这里使用 transform 得到一个新的列 'quantile'
        # 注意：groupby 的层级名称根据你的 DataFrame 而定，这里假设层级名称为 'time'
        df['quantile'] = df.groupby(level='time')[self.factor_col].transform(
            lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop')
        )

        # 根据时间和 quantile 分组，计算各层的平均未来收益率
        strat_ret = df.groupby(['time', 'quantile'])[self.ret_col].mean().unstack('quantile')

        # 计算累计收益率：假设 strat_ret 的每一行代表一个时间点的收益率，
        # 则累计收益率 = 累计乘积 (1 + 每日收益率) - 1
        cum_strat_ret = (1 + strat_ret).cumprod() - 1

        # 绘图
        plt.figure(figsize=(10, 4))
        for col in cum_strat_ret.columns:
            plt.plot(cum_strat_ret.index, cum_strat_ret[col], marker='o', label=f'Quantile {int(col) + 1}')
        plt.title(f"Cumulative Future Return by {quantiles}-Quantile Groups")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Future Return")
        plt.grid(True)
        plt.legend(title="Quantile")
        plt.show()


# ------------------ 示例用法 ------------------
if __name__ == "__main__":
    # 构造一个示例多重索引 DataFrame: index=(date, asset)，columns=[factor, future_return]
    with open('15min_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data.columns)
    data = data.rename(columns=str.lower)
    data = data.rename_axis(index=['time', 'asset'])
    asset = ['ONEUSDT', 'TRXUSDT', 'BTCUSDT', 'ICXUSDT', 'HOTUSDT', 'BANDUSDT', 'FTMUSDT', 'CHZUSDT',
             'VETUSDT', 'XTZUSDT', 'ONTUSDT', 'WAVESUSDT', 'BCHUSDT', 'DUSKUSDT', 'ZECUSDT', 'NEOUSDT',
             'QTUMUSDT', 'DASHUSDT', 'BATUSDT', 'IOTXUSDT', 'ETHUSDT', 'ANKRUSDT', 'ZRXUSDT', 'RVNUSDT',
             'DENTUSDT', 'OMGUSDT', 'IOSTUSDT', 'ENJUSDT', 'DOGEUSDT', 'COSUSDT', 'FETUSDT', 'IOTAUSDT',
             'ADAUSDT', 'RENUSDT', 'ALGOUSDT', 'XMRUSDT', 'ETCUSDT', 'TROYUSDT', 'KAVAUSDT', 'LINKUSDT',
             'NULSUSDT', 'NKNUSDT', 'XRPUSDT', 'RLCUSDT', 'XLMUSDT', 'HBARUSDT', 'BNBUSDT', 'MTLUSDT',
             'ZILUSDT']
    data = data[[a in asset for a in data.index.get_level_values('asset')]]
    data = data[-len(data) // 2:-len(data) // 4]
    print(data)
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-5) / x - 1).droplevel(0)
    data['returns'] = u.returns(data)
    data['vwap'] = u.vwap(data)

    data = data.dropna()
    evaluator = FactorEvaluator(data)
    # evaluator.plot_factor_distribution(factor_column='upsidevolumeratio', upper_bound=10, lower_bound=0)
    # evaluator.plot_cumulative_ic(factor_column='buysellratio', return_column='future_return',method='pearson')
    print(evaluator.hurst_exponent_for_multiindex(target_column='returns'))
