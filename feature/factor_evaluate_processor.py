import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter


class FactorEvaluator:
    def __init__(self, df: pd.DataFrame):
        """
        参数:
            df: 带有 MultiIndex (date, stock) 且包含 'factor' 和 'future_return' 两列的 DataFrame
        """
        self.df = df.sort_index()
        self.check_columns()

    def check_columns(self):
        required = ['factor', 'future_return']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def calc_ic(self) -> pd.Series:
        """
        计算 Spearman 信息系数（Rank IC），返回时间序列
        """

        def ic_per_day(group):
            return spearmanr(group['factor'], group['future_return'])[0]

        return self.df.groupby(level=0).apply(ic_per_day).rename("RankIC")

    def calc_layer_returns(self, n_layers=5) -> pd.DataFrame:
        """
        计算因子分层组合的平均收益
        参数:
            n_layers: 分层数量，默认为 5 层
        返回:
            各层的平均收益 DataFrame
        """
        df = self.df.copy()
        df['layer'] = df.groupby(level=0)['factor'].transform(
            lambda x: pd.qcut(x, n_layers, labels=False, duplicates='drop')
        )
        return df.groupby(['layer', df.index.get_level_values(0)])['future_return'].mean().unstack()

    def calc_mean_return_by_layer(self, n_layers=5) -> pd.Series:
        """
        返回每一层的平均收益（跨时间平均）
        """
        layered = self.calc_layer_returns(n_layers)
        return layered.mean(axis=1)

    def plot_factor_distribution(self):
        """
        可视化因子值分布（所有日期拼一起）
        """
        self.df['factor'].hist(bins=100)
        plt.title("Factor Value Distribution")
        plt.xlabel("Factor")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def calc_factor_return_series(self, n_layers=5) -> pd.Series:
        """
        返回 top 层（因子值最大）与 bottom 层（因子值最小）组合的收益差（日频）
        """
        layered = self.calc_layer_returns(n_layers)
        return (layered.loc[n_layers - 1] - layered.loc[0]).rename("Factor Return")

    def summary(self):
        """
        打印因子评估摘要信息
        """
        print("Mean Rank IC:", self.calc_ic().mean())
        print("IC Std:", self.calc_ic().std())
        print("Top - Bottom Return Mean:", self.calc_factor_return_series().mean())
        print("Top - Bottom Return Std:", self.calc_factor_return_series().std())

    def adf_test_each_stock(self, min_length=10, p_threshold=0.05) -> pd.DataFrame:
        """
        对每只资产的因子时间序列做ADF检验，返回带有统计量、p值、是否平稳的DataFrame
        """
        results = []
        for stock in self.df.index.get_level_values(1).unique():
            series = self.df.xs(stock, level=1)['factor'].dropna()
            if len(series) < min_length:
                continue
            stat, pval, _, _, crit, _ = adfuller(series)
            results.append({
                'stock': stock,
                'ADF Statistic': stat,
                'p-value': pval,
                'is_stationary': pval < p_threshold
            })
        return pd.DataFrame(results).set_index('stock')

    def stationary_ratio(self, min_length=10, p_threshold=0.05) -> float:
        """
        计算因子在个股时间序列上的平稳比率
        """
        adf_df = self.adf_test_each_stock(min_length, p_threshold)
        if len(adf_df) == 0:
            return np.nan
        return adf_df['is_stationary'].mean()

    def pairwise_cointegration_test(self, min_periods=50, p_threshold=0.05) -> pd.DataFrame:
        """
        对每对资产的因子时间序列进行 Engle-Granger 协整检验
        返回协整资产对及其 p 值
        """
        df_pivot = self.df['factor'].unstack()  # 变为 date x asset 表
        assets = df_pivot.columns
        results = []
        for a1, a2 in combinations(assets, 2):
            x = df_pivot[a1].dropna()
            y = df_pivot[a2].dropna()
            common_index = x.index.intersection(y.index)
            if len(common_index) < min_periods:
                continue
            score, pval, _ = coint(y.loc[common_index], x.loc[common_index])
            results.append({'asset1': a1, 'asset2': a2, 'p-value': pval})
        result_df = pd.DataFrame(results)
        return result_df.sort_values('p-value').reset_index(drop=True)

    '''
    trace_stats, crit_vals, asset_list = evaluator.johansen_cointegration_test()
    for i, asset in enumerate(asset_list):
        print(f"Trace stat {i+1}: {trace_stats[i]:.4f}, 95% crit: {crit_vals[i][1]}")
    '''

    def johansen_cointegration_test(self, det_order=0, k_ar_diff=1):
        """
        对所有资产因子时间序列进行 Johansen 多元协整检验
        返回:
            - trace statistics
            - 置信临界值
        """
        df_pivot = self.df['factor'].unstack().dropna()
        result = coint_johansen(df_pivot, det_order=det_order, k_ar_diff=k_ar_diff)
        trace_stats = result.lr1
        crit_vals = result.cvt
        return trace_stats, crit_vals, df_pivot.columns.tolist()

    def pca_analysis(self, n_components=3):
        """
        对所有资产因子时间序列进行 PCA 分析。
        参数:
            n_components: 提取的主成分数量
        返回:
            pca_model: sklearn PCA 对象（可用来查看解释方差等）
            factor_scores: DataFrame，index 为日期，columns 为 PC1, PC2...
        """
        df_pivot = self.df['factor'].unstack().dropna()  # date x asset
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_pivot)

        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(scaled)

        columns = [f'PC{i + 1}' for i in range(n_components)]
        factor_scores = pd.DataFrame(pcs, index=df_pivot.index, columns=columns)
        return pca, factor_scores

    def plot_layer_mean_return(self, n_layers=5):
        """
        画出各组的平均收益柱状图（分层回测结果）
        """
        mean_returns = self.calc_mean_return_by_layer(n_layers)
        mean_returns.plot(kind='bar', title=f'Mean Future Return by Factor Layer ({n_layers} layers)')
        plt.xlabel("Factor Layer (0 = lowest)")
        plt.ylabel("Mean Future Return")
        plt.grid(True)
        plt.show()

    def calc_top_bottom_return_diff(self, n_layers=5) -> float:
        """
        计算最高层与最低层的平均收益差
        参数:
            n_layers: 分组数量（例如 5 表示五分组）
        返回:
            float: top 层均值 - bottom 层均值
        """
        mean_ret = self.calc_mean_return_by_layer(n_layers)
        return float(mean_ret.iloc[-1] - mean_ret.iloc[0])

    def calc_top_bottom_return_series(self, n_layers=5, direction='top-bottom') -> pd.Series:
        """
        返回每期最高层与最低层之间的收益差（时间序列）
        参数:
            n_layers: 分层数量
            direction: 'top-bottom'（默认）或 'bottom-top'
        返回:
            pd.Series: 每期收益差序列
        """
        layered = self.calc_layer_returns(n_layers)

        if direction == 'top-bottom':
            ret_series = layered.loc[n_layers - 1] - layered.loc[0]
        elif direction == 'bottom-top':
            ret_series = layered.loc[0] - layered.loc[n_layers - 1]
        else:
            raise ValueError("direction must be either 'top-bottom' or 'bottom-top'.")

        return ret_series.rename(f"{direction.capitalize()} Return")

    def plot_top_bottom_cum_return(self, n_layers=5):
        """
        画出 Top-Bottom 收益差的累计收益曲线
        """
        diff_series = self.calc_top_bottom_return_series(n_layers)
        cum_return = diff_series.cumsum()
        cum_return.plot(title=f"Top-Bottom Cumulative Return ({n_layers} layers)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.show()

    def run_simple_backtest(self, n_layers=5, rf=0.0, plot=True, direction='top-bottom') -> pd.Series:
        """
        简易回测：构造多空组合，计算累计收益、夏普比率、最大回撤、卡玛比率
        参数:
            n_layers: 分层数
            rf: 年化无风险利率
            plot: 是否绘制累计收益图
            direction: 'top-bottom' 或 'bottom-top'
        返回:
            pd.Series: 每日多空组合收益
        """
        layered = self.calc_layer_returns(n_layers)

        if direction == 'top-bottom':
            ret_series = self.calc_top_bottom_return_series(n_layers, direction)
        elif direction == 'bottom-top':
            ret_series = self.calc_top_bottom_return_series(n_layers, direction)
        else:
            raise ValueError("direction 参数必须为 'top-bottom' 或 'bottom-top'")

        ret_series = ret_series.rename("LongShort Return")
        cum_ret = ret_series.cumsum()

        # 年化收益 & 波动率
        mean_ret = ret_series.mean()
        std_ret = ret_series.std()
        ann_return = mean_ret * 252
        ann_vol = std_ret * np.sqrt(252)
        sharpe = (mean_ret - rf / 252) / std_ret * np.sqrt(252)

        # 最大回撤
        cum_nav = (1 + ret_series).cumprod()
        peak = cum_nav.cummax()
        drawdown = 1 - cum_nav / peak
        max_drawdown = drawdown.max()

        # 卡玛比率
        calmar = ann_return / max_drawdown if max_drawdown > 0 else np.nan

        # 输出指标
        print(f"Strategy: {direction}")
        print(f"Annualized Return: {ann_return:.2%}")
        print(f"Annualized Volatility: {ann_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Calmar Ratio: {calmar:.2f}")

        # 画图
        if plot:
            cum_nav.plot(title=f"Backtest NAV: {direction}", figsize=(10, 4))
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (NAV)")
            plt.grid(True)
            plt.show()

        return ret_series

class FactorProcessor:
    def __init__(self, df: pd.DataFrame):
        """
        df: MultiIndex (date, asset) 的因子数据，要求包含 'factor' 列
        """
        self.df = df.copy()
        if 'factor' not in self.df.columns:
            raise ValueError("输入数据必须包含 'factor' 列")

    def denoise(self, method='ewma', window=5, inplace=False):
        """
        对因子时间序列进行去噪处理，支持 'ewma', 'sma', 'median', 'kalman'
        """
        df = self.df.copy()
        pivot = df['factor'].unstack()

        if method == 'sma':
            smoothed = pivot.rolling(window).mean()
        elif method == 'ewma':
            smoothed = pivot.ewm(span=window, adjust=False).mean()
        elif method == 'median':
            smoothed = pivot.rolling(window).median()
        elif method == 'kalman':
            smoothed = pd.DataFrame(index=pivot.index, columns=pivot.columns)
            for col in pivot.columns:
                series = pivot[col].dropna()
                if len(series) < 2:
                    continue
                kf = KalmanFilter(
                    transition_matrices=[1],
                    observation_matrices=[1],
                    initial_state_mean=series.iloc[0],
                    initial_state_covariance=1,
                    observation_covariance=1,
                    transition_covariance=0.01
                )
                state_means, _ = kf.filter(series.values)
                smoothed.loc[series.index, col] = state_means.ravel()
            smoothed = smoothed.astype(float)
        else:
            raise ValueError(f"不支持的方法：{method}")

        df['factor'] = smoothed.stack()
        if inplace:
            self.df['factor'] = df['factor']
        else:
            return df



