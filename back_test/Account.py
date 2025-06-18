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


class StopLossLogic(ABC):
    @abstractmethod
    def check_stop_loss(self, **kwargs):
        pass


class DefaultStopLossLogic(StopLossLogic):
    def __init__(self, max_drawdown=0.05):
        self.max_drawdown = max_drawdown
        self.highest_price_map = {}  # 用于多头跟踪最高价格
        self.lowest_price_map = {}  # 用于空头跟踪最低价格

    def init_holding(self, asset, price, **kwargs):
        if asset not in self.highest_price_map:
            self.highest_price_map[asset] = price
        else:
            self.highest_price_map[asset] = max(self.highest_price_map[asset], price)
        if asset not in self.lowest_price_map:
            self.lowest_price_map[asset] = price
        else:
            self.lowest_price_map[asset] = min(self.lowest_price_map[asset], price)

    def check_stop_loss(self, account, price_map, current_time, **kwargs):
        """
        这个函数似乎有问题
        """
        positions_to_close = []

        for (asset, direction), pos in account.positions.items():
            curr_price = price_map.get(asset, pos.entry_price)

            if direction == "long":
                # 多头逻辑：跟踪最高价格
                prev_high = self.highest_price_map.get(asset, pos.entry_price)
                new_high = max(prev_high, curr_price)
                self.highest_price_map[asset] = new_high
                # 如果当前价格低于最高价格的 (1 - max_drawdown)，止损
                if curr_price < new_high * (1 - self.max_drawdown):
                    positions_to_close.append((asset, direction))

            elif direction == "short":
                # 空头逻辑：跟踪最低价格
                prev_low = self.lowest_price_map.get(asset, pos.entry_price)
                new_low = min(prev_low, curr_price)
                self.lowest_price_map[asset] = new_low
                # 如果当前价格高于最低价格的 (1 + max_drawdown)，止损
                if curr_price > new_low * (1 + self.max_drawdown):
                    positions_to_close.append((asset, direction))

        return positions_to_close

    def holding_close(self, asset, **kwargs):
        del self.highest_price_map[asset]
        del self.lowest_price_map[asset]


class HoldNBarStopLossLogic(StopLossLogic):
    """
    持仓经过n根k线止盈止损
    """

    def __init__(self, windows=1):
        self.windows = windows
        self.holding_time = {}  # 用于跟踪持仓经过k线

    def init_holding(self, asset, **kwargs):
        self.holding_time[asset] = 0

    def check_stop_loss(self, account, current_time, **kwargs):
        """
        price_map为一个字典，为当前时间戳所有的资产信息
        """
        positions_to_close = []
        for (asset, direction), pos in account.positions.items():
            if asset not in self.holding_time:
                print(len(self.holding_time))
                print(len(account.positions))
                pass
            self.holding_time[asset] += 1
            if self.holding_time[asset] == self.windows:
                positions_to_close.append((asset, direction))
                self.holding_time[asset] = 0

        return positions_to_close

    def holding_close(self, asset, **kwargs):
        del self.holding_time[asset]


class CostThresholdStrategy(StopLossLogic):
    def __init__(self, gain_threshold=0.05, loss_threshold=0.02,windows=1):
        self.gain_threshold = gain_threshold
        self.loss_threshold = loss_threshold
        self.holding_cost = {}  # 用于跟踪持仓成本
        self.windows = windows
        self.holding_time = {}

    def init_holding(self, asset, direction, price, **kwargs):
        self.holding_cost[(asset, direction)] = price
        self.holding_time[asset] = 0

    def check_stop_loss(self, account, current_time, price_map, update_asset,**kwargs):
        """
        price_map为一个字df_asset = group_df.get_group('BTCUSDT')典，为当前时间戳所有的资产信息
        """
        positions_to_close = []
        for (asset, direction), pos in account.positions.items():
            curr_price = price_map.get(asset)
            if (asset, direction) not in self.holding_cost:
                raise ValueError(f"holding {asset} not exist")
            if asset == update_asset:
                self.holding_time[asset] += 1
            if self.holding_time[asset] == self.windows:
                positions_to_close.append((asset, direction))
                self.holding_time[asset] = 0
                continue
            if direction == "long":
                if curr_price < pos.entry_price * (1 - self.loss_threshold):
                    # 止损
                    positions_to_close.append((asset, direction))
                elif curr_price > pos.entry_price * (1 + self.gain_threshold):
                    # 止盈
                    positions_to_close.append((asset, direction))
            elif direction == "short":
                if curr_price > pos.entry_price * (1 + self.loss_threshold):
                    # 止损
                    positions_to_close.append((asset, direction))
                elif curr_price < pos.entry_price * (1 - self.gain_threshold):
                    positions_to_close.append((asset, direction))

        return positions_to_close

    def holding_close(self, asset, direction):
        del self.holding_cost[(asset, direction)]
        del self.holding_time[asset]

class CostATRStrategy(StopLossLogic):
    def __init__(self, long_gain_times=4, long_loss_times=2, short_gain_times=1, short_loss_times=1):
        self.holding_cost = {}  # 用于跟踪持仓成本
        self.long_gain_times = long_gain_times
        self.long_loss_times = long_loss_times
        self.short_gain_times = short_gain_times
        self.short_loss_times = short_loss_times

    def init_holding(self, asset, direction, price, **kwargs):
        self.holding_cost[(asset, direction)] = price

    def check_stop_loss(self, account, current_time, price_map, atr_value, **kwargs):
        """
        price_map为一个字典，为当前时间戳所有的资产信息
        """
        positions_to_close = []

        for (asset, direction), pos in account.positions.items():
            curr_price = price_map.get(asset)
            atr = atr_value[asset]
            if (asset, direction) not in self.holding_cost:
                raise ValueError(f"holding {asset} not exist")
            # print(pos.entry_price, self.loss_times * atr)
            if direction == "long":
                if curr_price < pos.entry_price - self.long_loss_times * atr:
                    # 止损
                    positions_to_close.append((asset, direction))
                elif curr_price > pos.entry_price + self.long_gain_times * atr:
                    # 止盈
                    positions_to_close.append((asset, direction))

            elif direction == "short":
                if curr_price < pos.entry_price - self.short_gain_times * atr:
                    # 止损
                    positions_to_close.append((asset, direction))
                elif curr_price > pos.entry_price + self.short_loss_times * atr:
                    # 止盈
                    positions_to_close.append((asset, direction))

        return positions_to_close

    def holding_close(self, asset, direction):
        del self.holding_cost[(asset, direction)]


class PositionManager:
    def __init__(self, threshold=0.05,fixed_pos = 0.05):
        self.threshold = threshold
        self.fixed_pos = fixed_pos
        pass

    def get_allocate_pos(self, asset, price, signal, market_cap, account: Account):
        """
        检测多空持仓，限制单方持仓大于6成仓位
        market_cap:权益市值(多+空)
        """
        long_cap = 0
        short_cap = 0
        for (asset, direction), pos in account.positions.items():
            if direction == "long":
                long_cap += pos.entry_price * pos.quantity
            elif direction == "short":
                short_cap += pos.entry_price * pos.quantity

        direct = "long" if signal == 1 else "short"
        asset_pos = 0
        if (asset, direct) in account.positions:
            asset_pos = account.positions[(asset, direct)].quantity * price

        total_cap = market_cap + account.cash
        if long_cap > total_cap * self.threshold and direct == "long":
            return 0
        if short_cap > total_cap * self.threshold and direct == "short":
            return 0
        target_pos = min((total_cap * self.threshold - asset_pos) * self.fixed_pos, account.cash)
        target_pos = target_pos if target_pos > 0 else 0
        return 100


class AtrPositionManager:
    def __init__(self, loss_times=1, risk_percent=0.01):
        self.holding_cost = {}  # 用于跟踪持仓成本
        self.loss_times = loss_times
        self.risk_percent = risk_percent

    def get_allocate_pos(self, asset, price, signal, market_cap, account: Account, atr_value):
        """
        检测多空持仓，限制单方持仓大于6成仓位
        """
        long_cap = 0
        short_cap = 0
        for (asset, direction), pos in account.positions.items():
            if direction == "long":
                long_cap += pos.entry_price * pos.quantity
            elif direction == "short":
                short_cap += pos.entry_price * pos.quantity

        direct = "long" if signal == 1 else "short"
        asset_pos = 0
        if (asset, direct) in account.positions:
            asset_pos = account.positions[(asset, direct)].quantity * price
        atr = atr_value[asset]
        total_cap = market_cap + account.cash

        risk_amount = total_cap * self.risk_percent
        per_unit_risk = atr * self.loss_times
        position_size = risk_amount / per_unit_risk * price
        target_pos = min(position_size, account.cash)
        if signal == 1:
            return target_pos
        elif signal == -1 or signal == -2:
            return target_pos
            # if short_cap / total_cap > 0.6:
            #     return 0
            # if short_cap / total_cap > 0.4:
            #     return target_pos / 2
            # else:
            #     return target_pos
        else:
            return 0
