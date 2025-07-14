import math
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd


def find_current_leverage_rate(rate_map: list, asset_name, current_time):
    """
    寻找当前资产的杠杆费率
    :param rate_map: 一个list类型，每个包含每种交易对的历史资金费率数据，为类型为pd.dataframe
    :param asset_name:资产名称
    :param current_time:当前时间
    :return:当前时间下对应的资产的杠杆费率
    """
    for leverage_history_df in rate_map:
        if leverage_history_df['asset'][0] != asset_name:
            continue
        position = leverage_history_df['time'].searchsorted(current_time)
        return leverage_history_df['rate'].iloc[position - 1]


def find_current_funding_rate(rate_map: list, asset_name, current_time) -> tuple:
    for leverage_history_df in rate_map:
        if leverage_history_df['asset'][0] != asset_name:
            continue
        if current_time not in leverage_history_df['time'].values:
            return False, -1
        else:
            # 确保返回的是单一的数值而不是 Series
            rate_value = leverage_history_df.loc[leverage_history_df['time'] == current_time, 'rate']
            if not rate_value.empty:
                return True, rate_value.iloc[0]
            return False, -1


class Account:
    """
    Account类只负责记录资金变化和交易记录
    """

    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.positions = {}
        self.transactions = []
        self.reversed_cash = 0

    def record_transaction(self, tx):
        self.transactions.append(tx)

    def print_positions(self):
        """
        打印所有持仓的详细参数
        """
        if not self.positions:
            print("No positions in the account.")
            return

        print("Positions:")
        for (asset, direction), position in self.positions.items():
            print(f"Asset: {position.asset}")
            print(f"Direction: {position.direction}")
            print(f"Quantity: {position.quantity}")
            print(f"Entry Price: {position.entry_price:.2f}")
            print(f"Leverage: {position.leverage:.2f}")
            print(f"Position Type: {position.position_type}")
            print("-" * 30)



class Position:
    """
    Position类对开仓的信息进行封装
    """

    def __init__(self, asset, direction, quantity, entry_price, leverage=1.0, position_type="spot"):
        """

        :param asset: 资产名称
        :param direction: 多/空方向
        :param quantity: 总数量
        :param entry_price: 开仓价格
        :param leverage: 杠杆倍率
        :param position_type:
        self.own_quantity 为自己持仓（不包含杠杆部分）市值

        """
        self.asset = asset
        self.direction = direction  # "long"/"short"
        self.quantity = quantity
        self.entry_price = entry_price
        self.leverage = leverage
        self.position_type = position_type  # "spot" or "future"

        # 自行扩展
        self.own_equity = entry_price * quantity / leverage  # 持仓市值（不包含杠杆的部分）


class LeverageManager:
    def __init__(self, rate_map: list, leverage: int):
        self.rate_map = rate_map
        self.leverage = leverage

    def settle_fees(self, account: Account, current_time: datetime, price_map: dict, is_open_close=False) -> None:
        if current_time.minute != 0 and current_time.second != 0 and is_open_close is False:  # 只有整小时或开仓关仓才会进行杠杆费用结算
            return
        for (asset, direction), pos in account.positions.items():
            #  遍历持仓，如果持有为现货，并且持仓的杠杆大于1.0 则进行费用的结算
            if pos.position_type == "spot" and pos.leverage > 1.0:
                hourly_rate = find_current_leverage_rate(self.rate_map, "USDT", current_time)

                current_price = price_map.get(asset, pos.entry_price)  # 获取当前价格
                notional_value = pos.quantity * current_price  # 获取总共市值
                borrowed_funds = notional_value - pos.own_equity  # 获取总共借贷的数值
                if borrowed_funds > 0:
                    fee = borrowed_funds * hourly_rate
                    account.cash -= fee
                    account.record_transaction({
                        "time": current_time,
                        "action": "leverage_fee",
                        "asset": asset,
                        "fee": fee
                    })


class FundingFeesManager:
    """
    类似于杠杆费率计算，但这是用于永续合约资金费率的计算。
    """

    def __init__(self, rate_map: list):
        self.rate_map = rate_map

    def settle_fees(self, account, current_time: datetime, price_map):
        for (asset, direction), pos in account.positions.items():
            if pos.position_type == "perpetual" and pos.leverage > 1.0:  # 确保只有永续合约和杠杆大于1.0的持仓进行费用结算
                is_found, rate = find_current_funding_rate(self.rate_map, asset, current_time)
                if not is_found:
                    continue
                current_price = price_map.get(asset, pos.entry_price)  # 使用当前价格或入场价格
                notional_value = pos.quantity * current_price  # 计算名义市值
                fee = notional_value * rate
                if direction == "long":
                    account.cash -= fee
                    account.record_transaction({
                        "time": current_time,
                        "action": "funding_fee",
                        "asset": asset,
                        "fee": -fee
                    })
                elif direction == "short":
                    account.cash += fee
                    account.record_transaction({
                        "time": current_time,
                        "action": "funding_fee",
                        "asset": asset,
                        "fee": fee
                    })
                else:
                    raise ValueError("Unexpected trade direction")




