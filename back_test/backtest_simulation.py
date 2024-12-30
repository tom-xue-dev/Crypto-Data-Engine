import math
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

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
        if self.account.cash < cost:
            print("Insufficient cash to open position.")
            return
        self.account.cash -= cost
        if leverage > 1:
            # 开仓计算杠杆 目前只支持做多杠杆计算
            if direction == "long":
                price_map = {asset: price}
                self.leverage_manager.settle_fees(self.account, current_time, price_map, is_open_close=True)

        pos = Position(asset, direction, quantity, price, leverage, position_type)
        self.account.positions[(asset, direction)] = pos
        self.account.record_transaction({
            "time": current_time,
            "action": "open",
            "asset": asset,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "leverage": leverage
        })

    def close_position(self, asset, direction, price, current_time):
        key = (asset, direction)
        if key not in self.account.positions:
            return
        pos = self.account.positions.pop(key)
        # 结算盈亏
        # 多头收益：quantity * (price - entry_price)
        # 这里略写
        if pos.leverage > 1:
            # 开仓计算杠杆 目前只支持做多杠杆计算
            if direction == "long":
                price_map = {asset: price}
                self.leverage_manager.settle_fees(self.account, current_time, price_map, is_open_close=True)

        if direction == "long":
            pnl = pos.quantity * (price - pos.entry_price)
        else:
            pnl = pos.quantity * (pos.entry_price - price)

        self.account.cash += (pos.own_equity + pnl)
        self.account.record_transaction({
            "time": current_time,
            "action": "close",
            "asset": asset,
            "direction": direction,
            "quantity": pos.quantity,
            "close_price": price,
            "pnl": pnl
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
            positions_to_close = self.stop_loss_logic.check_stop_loss(self.account, price_map, current_time)
            for (asset, direction) in positions_to_close:
                self.close_position(asset, direction, price_map.get(asset, 0), current_time)


class Backtest:
    """
    回测主流程：对每根bar做：
     1. 从row读取signal
     2. 调用 broker.open/close
     3. bar结束时，broker.on_bar_end -> 杠杆费 + 止损检查
    """

    def __init__(self, broker: Broker, strategy_results: list, pos_manager=PositionManager()):
        """
        :param broker: Broker对象，内部有account、leverage_manager、stop_loss_logic
        :param strategy_results: List[pd.DataFrame] 这里假设每个df对应一段时间(如每日/每小时)
        """
        self.broker = broker
        self.strategy_results = strategy_results
        self.pos_manager = pos_manager

    def run(self):
        for df in self.strategy_results:
            current_market_cap = self.get_market_cap(df)
            for idx, row in df.iterrows():
                asset = row['asset']
                price = row['close']
                signal = row['signal']
                current_time = row['time']
                # 1) 处理交易信号
                self.process_signal(signal, asset, price, current_time, current_market_cap)
                # 2) 在每行处理完后(可理解为每个 bar 结束)调用 on_bar_end
                price_map = {asset: price}  # 如果多资产，就把所有资产的价格都放进去
                self.broker.on_bar_end(current_time, price_map)

        return {
            "final_cash": self.broker.account.cash,
            "positions": self.broker.account.positions,
            "transactions": self.broker.account.transactions
        }

    def process_signal(self, signal, asset, price, current_time, current_market_cap):
        # 简化: signal=1 -> 开多, signal=-1 -> 开空, signal=0 -> 不操作
        position = self.pos_manager.get_allocate_pos(current_market_cap, self.broker.account.cash)
        quantity = position / price
        quantity = math.floor(quantity * 100) / 100  # 去尾法保证小数点后两位
        if quantity <= 0.01:  # 余额不足
            raise ValueError("insufficient amount cash to buy asset")

        if signal == 1:
            self.broker.open_position(
                asset=asset,
                direction="long",
                price=price,
                leverage=2.0,  # 或从别处读取
                current_time=current_time,
                position_type="spot",
                quantity=quantity  # 可以根据策略或资金管理计算
            )
        elif signal == -1:
            self.broker.open_position(
                asset=asset,
                direction="short",
                price=price,
                leverage=2.0,
                current_time=current_time,
                position_type="spot",
                quantity=quantity
            )
        else:
            pass  # signal=0, 不开仓

    def get_market_cap(self, current_df: pd.DataFrame):
        """
        计算当前持仓总市值（不包含现金）
        :param current_df:
        :return:
        """
        total_market_value = 0
        holdings = self.broker.account.positions
        for (asset, _), position in holdings.items():
            # 从DataFrame中获取当前价格
            if asset in current_df['asset'].values:
                # 从DataFrame中找到对应资产的当前价格
                current_price = current_df.loc[current_df['asset'] == asset, 'close'].iloc[0]
                total_market_value += position.quantity * current_price
        return total_market_value
