import datetime
import math
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Account import Account, Position, DefaultStopLossLogic, PositionManager


class Broker:
    """
    Broker 为撮合类，对回测交易的函数调用高层封装
    """

    def __init__(self, account: Account, leverage_manager=None, stop_loss_logic=None):
        """
        撮合订单的类
        传入账号据数据用于记录交易历史
        传入市场类型用于记录现货或者期货市场
        传入leverage_manager用于记录每小时末的杠杆费率结算
        传入stop_loss_logic用于进行止损的逻辑处理

        :param account:
        :param leverage_manager:
        :param stop_loss_logic:
        """
        self.account = account
        self.leverage_manager = leverage_manager
        self.stop_loss_logic = stop_loss_logic

    def open_position(self, asset, direction, quantity, price, leverage, position_type, current_time):
        # 简化: 扣除自有资金 = quantity * price / leverage
        cost = quantity * price / leverage
        # 计算手续费 现在默认是maker: 双向千1
        cost = cost * 1.001
        if self.account.cash < cost:
            print("Insufficient cash to open position.")
            return
        # print(f"open pos in {current_time}")
        self.account.cash -= cost
        if leverage > 1:
            # 开仓计算杠杆 目前只支持做多杠杆计算
            if direction == "long":
                price_map = {asset: price}
                self.leverage_manager.settle_fees(self.account, current_time, price_map, is_open_close=True)

        pos = Position(asset, direction, quantity, price, leverage, position_type)
        self.account.record_transaction({
            "time": current_time,
            "action": "open",
            "asset": asset,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "leverage": leverage
        })
        self.account.positions[(asset, direction)] = pos
        if self.stop_loss_logic:
            self.stop_loss_logic.init_holding(asset=asset, price=price, direction=direction)

    def close_position(self, asset, direction, price, current_time, stop_loss=False):
        key = (asset, direction)
        if key not in self.account.positions:
            raise ValueError(f"target asset{key} not in holdings")
        pos = self.account.positions.pop(key)
        if self.stop_loss_logic:
            self.stop_loss_logic.holding_close(asset=asset, direction=direction)
        # 结算盈亏
        # 多头收益：quantity * (price - entry_price)
        # 这里略写
        if pos.leverage > 1:
            # 开仓计算杠杆 目前只支持做多杠杆计算
            if direction == "long":
                price_map = {asset: price}
                self.leverage_manager.settle_fees(self.account, current_time, price_map, is_open_close=True)
        if pos.direction == "long":
            pnl = pos.quantity * (price - pos.entry_price)
        else:
            pnl = -pos.quantity * (price - pos.entry_price)
        total_gain = (pos.own_equity + pnl)
        self.account.cash += total_gain * 0.999  # 千分之一手续费
        self.account.record_transaction({
            "time": current_time,
            "action": "close",
            "asset": asset,
            "direction": direction,
            "quantity": pos.quantity,
            "close_price": price,
            "pnl": pnl,
            "stop_loss": stop_loss
        })

    def on_bar_end(self, current_time, price_map):
        """
        进行k线结束的一些检查 例如止损止盈检查，资金费率和杠杆费率结算等
        :param current_time:
        :param price_map:
        :return:
        """
        # 1) 杠杆结算
        if self.leverage_manager:
            self.leverage_manager.settle_fees(self.account, current_time, price_map)

        # 2) 止损检查
        if self.stop_loss_logic:

            positions_to_close = self.stop_loss_logic.check_stop_loss(account=self.account, price_map=price_map,
                                                                      current_time=current_time, holding=1)

            for (asset, direction) in positions_to_close:
                if price_map.get(asset) is not None:
                    if (asset, direction) not in self.account.positions:
                        raise ValueError(f"{asset},{direction} not found")
                    self.close_position(asset, direction, price_map.get(asset), current_time, stop_loss=True)
                else:
                    print(f"warning,cannot find price_map{asset, price_map.get(asset)}")


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

    def run(self):
        # 获取所有时间点
        all_times = self.strategy_results.index.get_level_values('time').unique()

        # 初始化存储净值历史
        net_value_history = []

        # 矢量化遍历时间
        for current_time in all_times:
            # 筛选当前时间的分组数据
            group_df = self.strategy_results.loc[current_time]
            # 矢量化计算当前市值（示例：假设为 `close` 总和）
            current_market_cap = self.get_market_cap(group_df)

            # 矢量化构建 price_map
            price_map = group_df['close'].to_dict()

            # 矢量化处理信号
            signals = group_df[['signal', 'close']].reset_index().to_dict('records')
            for signal_data in signals:
                self.process_signal(
                    signal_data['signal'],
                    signal_data['asset'],
                    signal_data['close'],
                    current_time,
                    current_market_cap
                )

            # 矢量化调用 on_bar_end
            if price_map:
                self.broker.on_bar_end(current_time, price_map)
            else:
                print(f"No prices found for time: {current_time}")

            # 矢量化记录当前净值
            self.log_net_value(group_df, current_time)

        return {
            "final_cash": self.broker.account.cash,
            "positions": self.broker.account.positions,
            "transactions": self.broker.account.transactions,
            "net_value_history": pd.DataFrame(self.net_value_history)
        }

    def process_signal(self, signal, asset, price, current_time, current_market_cap):
        # 简化: signal=1 -> 开多, signal=-1 -> 开空, signal=0 -> 不操作

        position = self.pos_manager.get_allocate_pos(current_market_cap, self.broker.account.cash)
        quantity = position / (price * 1.001)
        quantity = math.floor(quantity * 100) / 100  # 去尾法保证小数点后两位
        if self.broker.leverage_manager is not None:
            leverage = self.broker.leverage_manager.leverage
        else:
            leverage = 1  # 默认为1,不开杠杆
        if quantity <= 0.01:  # 余额不足
            # print(self.broker.account.cash, self.broker.account.positions)
            # raise ValueError("insufficient amount cash to buy asset")
            # print(f"insufficient cash,target price is {price}")
            return

        existing_long_key = (asset, "long")
        existing_short_key = (asset, "short")

        holdings = self.broker.account.positions  # 当前持仓 dict
        if signal == 1:
            if existing_long_key in holdings:
                return
            # if existing_short_key in holdings:
            #     self.broker.close_position(asset, "short", price, current_time)
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
            # if existing_long_key in holdings:
            #     self.broker.close_position(asset, "long", price, current_time)

            self.broker.open_position(
                asset=asset,
                direction="short",
                price=price,
                leverage=leverage,
                current_time=current_time,
                position_type="spot",
                quantity=quantity
            )
        else:
            # if existing_short_key in holdings:
            #     self.broker.close_position(asset, "short", price, current_time)
            # if existing_long_key in holdings:
            #     self.broker.close_position(asset, "long", price, current_time)
            pass  # signal=0, 不开仓

    def get_market_cap(self, current_df: pd.DataFrame):
        """
        计算当前持仓总市值（不包含现金）
        :param current_df: 包含当前市场数据的 DataFrame，需包含 'asset' 和 'close' 列
        :return: 持仓总市值
        """
        total_market_value = 0
        holdings = self.broker.account.positions

        price_map = dict(zip(current_df.index.get_level_values('asset'), current_df['close']))

        for (asset, direction), position in list(holdings.items()):
            if asset in price_map:
                current_price = price_map[asset]
                # 计算持仓总市值（绝对值）
                total_market_value += abs(position.quantity * current_price)
            else:
                # 如果找不到当前价格，抛出警告或记录日志,并且撤销买入操作
                # print(
                #     f"Warning: Current price for asset {asset} not found in data.,timestamp is {current_df.loc[0, 'time']}")
                print(f"warning: {asset} not found in data")
                self.broker.account.cash += abs(position.entry_price*position.quantity)
                del self.broker.account.positions[(asset,direction)]
                continue

        return total_market_value

    def log_net_value(self, current_df: pd.DataFrame, current_time: datetime.datetime):
        """
        计算并记录账户净值：
        1) 现金
        2) 所有持仓的市值
        """
        total_market_value = self.get_market_cap(current_df)
        net_value = self.broker.account.cash + total_market_value
        # 将当前时间、净值等信息保存
        self.net_value_history.append({
            "time": current_time,
            "net_value": round(net_value, 2),
            "cash": round(self.broker.account.cash, 2),
        })
