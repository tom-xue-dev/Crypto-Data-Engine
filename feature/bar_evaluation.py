import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Dataloader import DataLoader,DataLoaderConfig

class BarEvaluator:
    def __init__(self, df:pd.DataFrame):
        """
        :param df: DataFrame, Bar 数据, 必须包含 'close' 列
        """
        self.df = df

    def compute_volatility(self):
        returns = self.df['close'].pct_change().dropna()
        return returns.std()

    def compute_autocorrelation(self):
        returns = self.df['close'].pct_change().dropna()
        return returns.autocorr()

    def plot_return_distribution(self):
        returns = self.df['close'].pct_change().dropna()
        print(f"Skewness: {returns.skew()}")
        print(f"Kurtosis: {returns.kurtosis()}")
        plt.figure(figsize=(10, 4))
        plt.hist(returns, bins=100, alpha=0.7)
        plt.title(f"{self.df.index.get_level_values('asset')[0]}Return Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.show()

    def simple_signal_backtest(self, signal_column="vwap"):
        """
        简易信号测试：价格突破 VWAP 时开仓，下一 Bar 平仓
        """
        signals = np.where(self.df['close'] > self.df[signal_column], 1, -1)
        returns = self.df['close'].pct_change().shift(-1)
        strategy_returns = signals[:-1] * returns[:-1]
        total_return = np.sum(strategy_returns)
        volatility = np.std(strategy_returns)
        sharpe = total_return / volatility if volatility != 0 else np.nan

        print(f"策略总收益: {total_return:.4f}")
        print(f"策略波动率: {volatility:.4f}")
        print(f"夏普比: {sharpe:.2f}")

    def evaluate(self):
        print("========== Bar 评估 ==========")
        print(f"Volatility: {self.compute_volatility():.6f}")
        print(f"Autocorrelation: {self.compute_autocorrelation():.4f}")
        print("\n========== 简易信号回测 ==========")
        self.simple_signal_backtest()
        print("\n========== 收益分布图 ==========")
        self.plot_return_distribution()

class BarQualityScanner:
    def __init__(self, data):
        """
        :param data: MultiIndex DataFrame, index=['time', 'asset'], 包含 'close' 列
        """
        self.data = data

    def evaluate_asset(self, asset_df):
        returns = asset_df['close'].pct_change().dropna()
        volatility = returns.std()
        autocorr = returns.autocorr()
        kurtosis = returns.kurtosis()
        skewness = returns.skew()

        recommendation = "✅ 正常"
        if volatility > 0.06 or abs(autocorr) > 0.1 or kurtosis > 20 or abs(skewness) > 1:
            recommendation = "❗️建议调高 threshold"

        return volatility, autocorr, kurtosis, skewness, recommendation

    def scan(self):
        results = []
        for asset, group in self.data.groupby('asset'):
            vol, auto, kurt, skew, rec = self.evaluate_asset(group)
            results.append({
                'asset': asset,
                'volatility': vol,
                'autocorrelation': auto,
                'kurtosis': kurt,
                'skewness': skew,
                'recommendation': rec
            })

        result_df = pd.DataFrame(results)
        return result_df

if __name__ == "__main__":
    config = DataLoaderConfig.load("load_config.yaml")
    data_loader = DataLoader(config)
    data = data_loader.load_all_data()
    for asset,group in data.groupby('asset'):
        evaluator = BarEvaluator(group)
        evaluator.plot_return_distribution()

