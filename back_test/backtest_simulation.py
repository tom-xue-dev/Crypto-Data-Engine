import datetime
import math
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Account import Account, Position, DefaultStopLossLogic, PositionManager



class Backtest:
    """
    回测主流程：对每根bar做：
     1. 从row读取signal
     2. 调用 broker.open/close
     3. bar结束时，broker.on_bar_end -> 杠杆费 + 止损检查
    """

    def __init__(self, broker: Broker, strategy_results: pd.DataFrame, pos_manager=PositionManager()):
        """
        :param broker: Broker对象，内部有account、leverage_manager、stop_loss_logic
        :param strategy_results: List[pd.DataFrame] 这里假设每个df对应一段时间(如每日/每小时)
        """
        self.broker = broker
        self.strategy_results = strategy_results
        self.pos_manager = pos_manager
        self.net_value_history = []
        self.cross_section = True

    def run(self):
        """
        最高效：一次顺序扫描 MultiIndex(time, asset)。
        适用于绝大多数 time 只有 1 行、少数多行的场景。
        """
        df = self.strategy_results.sort_index(level='time')

        price_map = {}
        current_time = None
        current_market_cap = 0
        cnt = 0

        for index_tuple, close, signal in zip(df.index, df['close'].values, df['signal'].values):
            time_i, asset = index_tuple  # MultiIndex 拆分
            price = close
            if time_i != current_time:
                if current_time is not None:
                    self.log_net_value(price_map, current_time)
                current_time = time_i
                if cnt % 10_000 == 0:
                    print(current_time)
                cnt += 1
                current_market_cap = self.get_market_cap(price_map)
            price_map[asset] = price
            self.broker.on_bar_end(current_time, price_map, asset)
            self.process_signal(
                signal=signal,
                asset=asset,
                price=price,
                current_time=current_time,
                current_market_cap=current_market_cap,
            )
        if current_time is not None:
            self.log_net_value(price_map, current_time)
        self.broker.clear_position(price_map)

        return {
            "final_cash": self.broker.account.cash,
            "positions": self.broker.account.positions,
            "transactions": self.broker.account.transactions,
            "net_value_history": pd.DataFrame(self.net_value_history)
        }

    def process_signal(self, signal, asset, price, current_time, current_market_cap, atr_value=None):
        # 简化: signal=1 -> 开多, signal=-1 -> 开空, signal=0 -> 不操作

        position = self.pos_manager.get_allocate_pos(asset, price, signal, current_market_cap, self.broker.account)
        quantity = position / (price * (1+self.broker.fees))
        quantity = math.floor(quantity * 100) / 100  # 去尾法保证小数点后两位

        if self.broker.leverage_manager is not None:
            leverage = self.broker.leverage_manager.leverage
        else:
            leverage = 1  # 默认为1,不开杠杆
        existing_long_key = (asset, "long")
        existing_short_key = (asset, "short")

        holdings = self.broker.account.positions  # 当前持仓 dict
        if signal == 1:
            if existing_long_key in holdings:
                return
            if existing_short_key in holdings:
                self.broker.close_position(asset, "short", price, current_time)
            if quantity <= 0.001:  # 余额不足
                return
            # print(f"long {asset} at",current_time)
            self.broker.open_position(
                asset=asset,
                direction="long",
                price=price,
                leverage=leverage,  # 或从别处读取
                current_time=current_time,
                position_type="spot",
                quantity=quantity  # 可以根据策略或资金管理计算
            )
        elif signal == -1:
            if existing_short_key in holdings:
                return
            if existing_long_key in holdings:
                self.broker.close_position(asset, "long", price, current_time)
            if quantity <= 0.01:  # 余额不足
                return
            # print(f"short {asset} at", current_time)
            self.broker.open_position(
                asset=asset,
                direction="short",
                price=price,
                leverage=leverage,
                current_time=current_time,
                position_type="spot",
                quantity=quantity
            )
        # else:
        #     if existing_long_key in holdings:
        #         self.broker.close_position(asset, "long", price, current_time)
        #     if existing_short_key in holdings:
        #         self.broker.close_position(asset, "short", price, current_time)

    def get_market_cap(self, price_map: dict):
        """
        计算当前持仓总市值（不包含现金）
        :param price_map: 当前最新价格字典，格式为 {asset: price}
        :return: 当前总市值
        """
        total_market_value = 0
        holdings = self.broker.account.positions
        long_cap = short_cap = 0
        for (asset, direction), position in list(holdings.items()):
            if asset in price_map:
                current_price = price_map[asset]
                if direction == "long":
                    long_cap += current_price * position.quantity
                else:
                    short_cap += current_price * position.quantity
            else:
                # 缺失价格时警告，但不删除持仓
                print(f"Warning: Missing price for asset {asset}, cannot compute market cap at this time.")
                continue
        total_market_value = long_cap + 2 * self.broker.account.reversed_cash-short_cap
        return total_market_value

    def log_net_value(self, price_map:dict, current_time: datetime.datetime):
        """
        计算并记录账户净值：
        1) 现金
        2) 所有持仓的市值
        """
        total_market_value = self.get_market_cap(price_map)
        net_value = self.broker.account.cash + total_market_value
        # 将当前时间、净值等信息保存
        self.net_value_history.append({
            "time": current_time,
            "net_value": round(net_value, 2),
            "cash": round(self.broker.account.cash, 2),
        })
