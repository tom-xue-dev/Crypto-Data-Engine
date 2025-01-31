import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class PerformanceAnalyzer:
    def __init__(self, net_value_df: pd.DataFrame, asset_name: str = None, asset_data: pd.DataFrame = None):
        """
        net_value_df: DataFrame，至少包含 ['time', 'net_value'] 字段。
        时间可以是索引或者一列，根据你的数据格式定。
        """
        self.asset_name = asset_name
        if not asset_name is None:
            if asset_name not in asset_data.index.get_level_values('asset'):
                raise ValueError(f"指定的资产名称 '{asset_name}' 不在数据中。")
            self.asset_net_value_df = asset_data.loc[(slice(None), asset_name), :].reset_index(level='asset', drop=True)

        self.net_value_df = net_value_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """
        将 time 转换为 datetime，并按时间排序等预处理。
        也可以在这里计算日度收益率、累计收益等。
        """
        # 如果 time 不是 datetime，需要先转换
        if not np.issubdtype(self.net_value_df['time'].dtype, np.datetime64):
            self.net_value_df['time'] = pd.to_datetime(self.net_value_df['time'])

        # 按时间排序
        self.net_value_df.sort_values(by='time', inplace=True)

        # 设置 time 为索引（可选）
        self.net_value_df.set_index('time', inplace=True)

        # 计算收益率 (如按 bar 计算)
        self.net_value_df['returns'] = self.net_value_df['net_value'].pct_change().fillna(0)

        self.net_value_df['normalized_net_value'] = (
                self.net_value_df['net_value'] / self.net_value_df['net_value'].iloc[0]
        )
        if self.asset_name:
            self.asset_net_value_df['normalized_net_value'] = (
                    self.asset_net_value_df['close'] / self.asset_net_value_df['close'].iloc[0]
            )

    def plot_net_value(self):
        """
        绘制净值曲线
        """
        # self.net_value_df, self.asset_net_value_df = self.net_value_df.align(
        #     self.asset_net_value_df, join='inner'
        # )
        plt.figure(figsize=(10, 6))

        plt.plot(self.net_value_df.index, self.net_value_df['normalized_net_value'], label='Account Net Value')
        if self.asset_name:
            plt.plot(self.net_value_df.index, self.asset_net_value_df['normalized_net_value'],
                     label=f'{self.asset_name} Net Value')
        plt.title('Net Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Net Value')
        plt.legend()
        plt.show()

    def calculate_max_drawdown(self) -> float:
        """
        计算最大回撤
        """
        cum_max = self.net_value_df['net_value'].cummax()
        drawdown = self.net_value_df['net_value'] / cum_max - 1
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_annual_return(self, annual_factor: int = 365) -> float:
        """
        计算年化收益率（假设每年有 annual_factor 个交易日/交易bar，可根据实际情况调整）
        """
        # 先计算累计收益率
        final_net_value = self.net_value_df['net_value'].iloc[-1]
        init_net_value = self.net_value_df['net_value'].iloc[0]
        total_return = final_net_value / init_net_value - 1

        # 计算回测总天数
        total_days = (self.net_value_df.index[-1] - self.net_value_df.index[0]).days
        if total_days == 0:
            return 0.0

        # 年化系数（粗略）
        yearly_periods = total_days / 365
        annual_return = (1 + total_return) ** (1 / yearly_periods) - 1
        return annual_return

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0, annual_factor: int = 365) -> float:
        """
        计算夏普比率:
        Sharpe = (Mean(returns) - risk_free_rate) / Std(returns) * sqrt(annual_factor)
        """
        rets = self.net_value_df['returns']
        mean_ret = rets.mean()
        std_ret = rets.std()
        if std_ret == 0:
            return 0.0
        sharpe = (mean_ret - risk_free_rate / annual_factor) / std_ret * np.sqrt(annual_factor)
        return sharpe

    def summary(self):
        """
        输出常见指标的汇总信息
        """
        max_dd = self.calculate_max_drawdown()
        ann_ret = self.calculate_annual_return()
        sharpe = self.calculate_sharpe_ratio()

        print("Performance Summary:")
        print(f"Final Net Value: {self.net_value_df['net_value'].iloc[-1]:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Annual Return: {ann_ret:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
