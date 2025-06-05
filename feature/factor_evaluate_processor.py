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
import Dataloader as dl
from Factor import *


class FactorEvaluator:
    def __init__(
            self,
            df: pd.DataFrame,
            factor_col: str = 'factor',
            future_return_col: str = 'future_return',
            n_future_days: int = 0
    ):
        """
        参数:
            df: MultiIndex (date, stock) 的 DataFrame
            factor_col: 因子列名称，默认 'factor'
            future_return_col: 未来收益列名称，默认 'future_return'
            n_future_days: 若 > 0，并且 future_return_col 不在 df 中，
                           则自动根据 'close' 列计算未来 n 天收益
        """
        # 1) 先对 df 排序、复制，防止原数据被改动
        self.df = df.sort_index().copy()
        self.factor_col = factor_col
        self.future_return_col = future_return_col

        # 2) 如果用户指定了 n_future_days > 0，但没提供 future_return_col，
        #    并且 df 中含有 'close' 列，则自动生成
        if n_future_days > 0 and self.future_return_col not in self.df.columns:
            if 'close' not in self.df.columns:
                raise ValueError(
                    "No 'close' column found. Cannot auto-generate future n-day returns."
                )
            print(f"[INFO] Auto-generating {n_future_days}-day future returns as '{self.future_return_col}'.")
            print(self.df)
            print(self.df.index.names)
            print(self.df.index.is_unique)
            dupe_mask = self.df.index.duplicated(keep=False)
            df_duplicates = self.df[dupe_mask]
            print(df_duplicates)
            df_unique = df[~df.index.duplicated(keep='first')]
            print(df_unique.index.names)
            self.df[self.future_return_col] = df_unique.groupby(level='asset')['close'].apply(
                lambda x: x.shift(-n_future_days) / x - 1
            ).droplevel(0)

        # 3) 核对是否包含必要列
        self.check_columns()

    def check_columns(self):
        required = [self.factor_col, self.future_return_col]
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

    def time_series_ic(df_single_asset):
        """
        df_single_asset：index=date, columns=['factor','future_return']
        返回：一个标量——整个时间序列上的相关系数
        """
        # 注意先对齐/shift, factor(t) 对应 future_return(t+1)?
        # 具体你可以视需求改
        return spearmanr(df_single_asset['factor'], df_single_asset['future_return'])[0]

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
        self.df[self.factor_col].hist(bins=100)
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
            series = self.df.xs(stock, level=1)[self.factor_col].dropna()
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
        df_pivot = self.df[self.factor_col].unstack()  # 变为 date x asset 表
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
        df_pivot = self.df[self.factor_col].unstack().dropna()
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
        df_pivot = self.df[self.factor_col].unstack().dropna()  # date x asset
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

    def backtest_each_asset_quantile(
            self,
            long_range=(0.1, 0.5),
            short_range = (0.1, 0.5),
            window=60,
            plot_nav=True,
            is_long=True,
            is_short=True
    ):
        """
        基于时间序列因子分位数阈值来决定多空头寸。
        对每支资产进行独立回测，汇总各资产表现，并可选画出净值曲线。
        做多:medium到high的因子
        做空:low到medium的因子
        参数:
            high_q: 做多因子阈值
            low_q: 做空因子阈值
            window: 滚动窗口
            plot: 是否绘制年化收益分布
            plot_nav: 是否绘制每只资产的净值曲线
            nav_limit: 最多绘制多少个资产净值（避免图太乱）

        返回:
            pd.DataFrame: 每个资产的回测指标
        """
        results = []
        nav_dict = {}  # 存每只资产的净值曲线
        all_assets = self.df.index.get_level_values(1).unique()

        for asset in all_assets:
            df_asset = self.df.xs(asset, level=1).copy()
            if len(df_asset) < window:
                continue

            factor_series = df_asset[self.factor_col]
            ret_series = df_asset[self.future_return_col]

            df_asset['position'] = 0
            mask_long = (factor_series > long_range[0]) & (factor_series < long_range[1])
            mask_short = (factor_series > short_range[0]) & (factor_series < short_range[1])
            if is_long:
                df_asset.loc[mask_long, 'position'] = 1
            if is_short:
                df_asset.loc[mask_short, 'position'] = -1

            df_asset['strategy_ret'] = df_asset['position'] * ret_series
            strategy_ret = df_asset['strategy_ret'].fillna(0)

            years = df_asset.index[-1] - df_asset.index[0]
            year = years.days / 365

            cum_nav = (1 + strategy_ret).cumprod()
            ann_return = np.e ** (np.log(cum_nav.iloc[-1]) / year) - 1
            ann_vol = strategy_ret.std() * (365 ** 0.5)
            sharpe = ann_return / ann_vol if ann_vol != 0 else None

            # 改为正向计算最大回撤
            max_dd = self.calc_max_drawdown_area(strategy_ret)
            kalma = ann_return / max_dd
            print(ann_return, max_dd)

            results.append({
                "asset": asset,
                "annual_return": ann_return,
                "annual_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "kalma": kalma
            })

            # 保存净值
            if plot_nav:
                nav_dict[asset] = cum_nav

        result_df = pd.DataFrame(results).set_index('asset')

        return result_df, nav_dict

    def plot_nvalue(
            self,
            long_range = (0.1, 0.5),
            short_range = (0.1, 0.5),
            window=60,
            nav_limit=100,  # 最多画多少条净值线
            is_long=True,
            is_short=True):
        strategy_value, nav_dict = self.backtest_each_asset_quantile(
            long_range=long_range,
            short_range= short_range,
            window=window,
            is_long=is_long,
            is_short=is_short)
        bench_mark, bench_mark_dict = self.backtest_each_asset_quantile(
            long_range=(-np.inf, np.inf),
            window=60,
            is_long=True,
            is_short=False)

        print("strategy:", strategy_value)
        print("benchmark:", bench_mark)
        comparison = strategy_value['annual_return'] > bench_mark['annual_return']
        proportion = comparison.mean()
        print(f"跑赢基准的资产的比例为: {proportion:.2%}")

        for asset, value in nav_dict.items():
            bench_mark_value = bench_mark_dict[asset]
            plt.plot(value.index, value, color="red", label="strategy_value")
            plt.plot(value.index, bench_mark_value, color="blue", label="bench_mark value")
            plt.xlabel('time')
            plt.ylabel('net value')
            plt.title(f'{asset} strategy value and bench mark value')
            plt.legend()
            plt.show()

        # return result_df

    def calc_max_drawdown_area(self, strategy_ret: pd.Series) -> float:
        """
        计算最大回撤面积（最大回撤区段内的累计回撤和）
        """
        cum_nav = (1 + strategy_ret).cumprod()
        peak = cum_nav.cummax()
        drawdown = 1 - cum_nav / peak  # 正值越大表示回撤越深

        return np.max(drawdown)


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

    def rolling_winsorize(self, window=60, lower=0.01, upper=0.99, inplace=False):
        """
        针对每支资产的时间序列做滚动分位去极值，避免未来泄露
        """
        df = self.df.copy()

        def _rolling_clip(x):
            return x.rolling(window).apply(
                lambda s: np.clip(s.iloc[-1], s.quantile(lower), s.quantile(upper)), raw=False
            )

        df['factor'] = df.groupby(level=1)['factor'].transform(_rolling_clip)
        if inplace:
            self.df['factor'] = df['factor']
        else:
            return df

    def rolling_standardize(self, window=60, method='zscore', inplace=False):
        """
        针对每支资产的时间序列做滚动标准化（z-score）
        """
        df = self.df.copy()
        if method == 'zscore':
            def _z(x):
                mean = x.rolling(window).mean()
                std = x.rolling(window).std()
                return (x - mean) / std

            df['factor'] = df.groupby(level=1)['factor'].transform(_z)
        elif method == 'rank':
            df['factor'] = df.groupby(level=1)['factor'].transform(
                lambda x: x.rolling(window).apply(lambda s: s.rank(pct=True).iloc[-1])
            )
        else:
            raise ValueError("method must be 'zscore' or 'rank'")
        if inplace:
            self.df['factor'] = df['factor']
        else:
            return df
        for i in range(10):


if __name__ == '__main__':
    alpha_funcs = [
        'alpha3',
    ]
    config = dl.DataLoaderConfig.load("load_config.yaml")
    data_loader = dl.DataLoader(config)
    df = data_loader.load_all_data()
    print(df)
    FC = FactorConstructor(df)
    FC.run_alphas(alpha_funcs)
    FE = FactorEvaluator(df, "alpha3", n_future_days=1)
    FE.plot_factor_distribution()
    long_range = (-np.inf,-0.05)
    short_range = (0.05,np.inf)
    FE.plot_nvalue(long_range, short_range,is_long=True,is_short=False)
