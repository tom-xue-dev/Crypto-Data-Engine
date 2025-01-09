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
    def __init__(self, dataset: list, asset: list):
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


class DualMAStrategy(Strategy):
    def __init__(self, dataset: list, asset: list, long: int, short: int):
        """
        双均线策略
        :param dataset:
        :param asset:
        :param long: 长周期
        :param short:短周期
        """
        super().__init__(dataset, asset)
        self.long_period = long
        self.short_period = short
        self.calculate_MA(self.long_period)
        self.calculate_MA(self.short_period)

    def calculate_MA(self, period):
        """
        计算MA均线,以收盘价为例
        :return:
        """
        full_df = pd.concat(self.dataset, ignore_index=True)
        # 确保数据按 time 排序（如果 time 可以转为 datetime 更好）
        full_df['time'] = pd.to_datetime(full_df['time'])
        full_df = full_df.sort_values(['asset', 'time'])
        # 按 name 分组，对 close 列进行 rolling mean
        full_df[f'MA{period}'] = full_df.groupby('asset')['close'].transform(
            lambda x: x.rolling(period).mean())
        df = full_df
        grouped = df.groupby('time')
        self.dataset = [group.reset_index(drop=True) for _, group in grouped]

    def generate_signal(self) -> None:
        """
        生成信号，默认短期MA上穿长期时用收盘价开多，信号为1
        反之开空 信号为-1
        :return: None
        """

        for index, time_frame_df in enumerate(self.dataset):
            if index < self.long_period:
                time_frame_df['signal'] = 0
                continue
            prev_df = self.dataset[index - 1]
            # 开多的情况
            long_condition = (prev_df[f'MA{self.short_period}'] < prev_df[f'MA{self.long_period}']) & (
                    time_frame_df[f'MA{self.short_period}'] >= time_frame_df[f'MA{self.long_period}'])
            # 开空的情况
            short_condition = (prev_df[f'MA{self.short_period}'] > prev_df[f'MA{self.long_period}']) & (
                    time_frame_df[f'MA{self.short_period}'] <= time_frame_df[f'MA{self.long_period}'])
            # 初始化 signal 列为 0
            time_frame_df['signal'] = 0
            # 更新满足条件的信号
            time_frame_df.loc[long_condition, 'signal'] = 1
            time_frame_df.loc[short_condition, 'signal'] = -1

        return


if __name__ == "__main__":
    start_time = "2023-12-01"
    end_time = "2024-6-30"
    asset_list = select_assets(spot=True, n=150)
    day_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "1d")
    strategy = DualMAStrategy(dataset=day_data_list, asset=asset_list, long=50, short=5)
