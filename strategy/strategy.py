import sys

import pandas as pd

from abc import abstractmethod
from datetime import datetime, timedelta
import numpy as np

from back_test.read_large_files import select_assets, load_filtered_data_as_list


def days_to_quarterly_settlement(date):
    # 获取当前日期的月份
    month = date.month
    year = date.year

    # 确定当前季度的最后月份
    if month <= 3:
        last_month = 3
    elif month <= 6:
        last_month = 6
    elif month <= 9:
        last_month = 9
    else:
        last_month = 12

    # 获取该季度最后一个月的最后一天
    last_day_of_quarter = datetime(year, last_month, 1) + timedelta(days=31)
    last_day_of_quarter = last_day_of_quarter.replace(day=1) - timedelta(days=1)

    # 找到该月的最后一个周五
    while last_day_of_quarter.weekday() != 4:
        last_day_of_quarter -= timedelta(days=1)

    # 计算剩余天数
    return (last_day_of_quarter - date).days


class Strategy:
    def __init__(self, dataset: pd.DataFrame, asset: list):
        """
        初始化具体策略，可以在此定义一些默认参数或初始化逻辑。
        """
        self.parameters = {}
        self.dataset = dataset
        self.assets_names = asset
        print("Strategy initialized.")

    def get_dataset(self):
        return self.dataset

    def init(self, **kwargs):
        """
        设置或更新策略参数。

        参数:
            kwargs: 字典形式的参数，例如阈值、系数等。
        """
        self.parameters.update(kwargs)
        print("Parameters set:", self.parameters)

    @abstractmethod
    def generate_signal(self) -> pd.DataFrame:
        """
        generate trading signals.
        :return: a dataframe whose indexes include 'time','price_num','signal'
        time:datetime
        price_num is the price of Nth asset,
        signal is a bitmap string which represent the operations for each asset.
        """


class BasisArbitrageStrategy(Strategy):
    def __init__(self, spot_data: pd.DataFrame, future_data: pd.DataFrame, asset_names: list):
        """
        :param spot_data:
        :param future_data:
        :param asset_names: the list of the asset, e.g. [BTC,ETH]
        """
        super().__init__()
        try:
            self.dataset = pd.merge(spot_data, future_data, on='time')
        except (ValueError, KeyError) as e:
            print("数据集合并错误，检查数据格式是否对齐，time 是否都为 datetime 格式")
        self.assets_names = asset_names
        print(f"index_len = {spot_data.columns}")
        zeros = '0' * len(self.assets_names) * 2
        print(zeros)
        self.dataset['signal'] = zeros  # 初始化信号
        print(self.dataset['signal'])

    def calculate_expected_return(self, spot_price, future_price, time):
        """
        calculate the expected return after long and short the pair asset until the delivery date.
        :param time:
        :param spot_price:
        :param future_price:
        :return:
        """
        profit = abs(future_price - spot_price)
        profit_ratio = profit / ((future_price + spot_price) / 2)  # 假设收敛到中间
        return_ratio = profit_ratio / timedelta(time).total_seconds() / (8 * 3600)
        return return_ratio

    def generate_signal(self):
        # Condition 1: Arbitrage opportunity for opening long on A and short on B
        self.dataset['DTS'] = self.dataset['date'].apply(days_to_quarterly_settlement)
        print(self.dataset['DTS'])


class DualMAStrategy():
    def __init__(self, dataset: pd.DataFrame, long_period: int, short_period: int):
        """
        双均线策略（重构版本）
        - dataset: 包含 [time, asset, open, high, low, close, ...] 等列的数据
        - long_period: 长周期
        - short_period: 短周期
        """
        self.dataset = dataset
        self.long_period = long_period
        self.short_period = short_period

        # 计算均线并生成交易信号
        self.calculate_MA()
        #self.generate_signal()

    def calculate_MA(self):
        """
        计算MA均线, 以收盘价为例
        将数据按照 ['asset','time'] 排序并设置为 MultiIndex，便于分组滚动计算
        """
        # 将数据按照 asset、time 排序，并设置为多级索引
        # self.dataset.sort_values(['asset', 'time'], inplace=True)
        # self.dataset.set_index(['time', 'asset'], inplace=True)

        # 分资产滚动计算长短均线
        self.dataset[f'MA{self.long_period}'] = (
            self.dataset
            .groupby(level='asset')['close']
            .rolling(self.long_period)
            .mean()
            .values
        )
        self.dataset[f'MA{self.short_period}'] = (
            self.dataset
            .groupby(level='asset')['close']
            .rolling(self.short_period)
            .mean()
            .values
        )
    def generate_signal(self):
        """
        生成信号：
        - 短期MA上穿长期MA --> 开多 signal = 1
        - 短期MA下穿长期MA --> 开空 signal = -1
        - 其余情况 signal = 0
        """
        def _signal_generation(df):
            # 复制一份，避免对原 DataFrame 产生副作用
            df = df.copy()
            df['signal'] = 0

            # 上一根K线短期MA < 长期MA，当前短期MA >= 长期MA --> 做多
            cond_long = (
                (df[f'MA_{self.short_period}'].shift(1) < df[f'MA_{self.long_period}'].shift(1)) &
                (df[f'MA_{self.short_period}'] >= df[f'MA_{self.long_period}'])
            )

            # 上一根K线短期MA > 长期MA，当前短期MA <= 长期MA --> 做空
            cond_short = (
                (df[f'MA_{self.short_period}'].shift(1) > df[f'MA_{self.long_period}'].shift(1)) &
                (df[f'MA_{self.short_period}'] <= df[f'MA_{self.long_period}'])
            )

            df.loc[cond_long, 'signal'] = 1
            df.loc[cond_short, 'signal'] = -1

            return df

        # 按资产分组，然后应用信号生成逻辑
        self.dataset = (
            self.dataset
            .groupby(level='asset', group_keys=False)
            .apply(_signal_generation)
        )

    def get_dataset(self) -> pd.DataFrame:
        """
        返回带有均线及信号列的结果 DataFrame
        """
        return self.dataset


if __name__ == "__main__":
    start_time = "2023-12-01"
    end_time = "2024-6-30"
    asset_list = select_assets(spot=True, n=150)
    day_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "1d")
    strategy = DualMAStrategy(dataset=day_data_list, asset=asset_list, long=50, short=5)
