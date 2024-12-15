import math
from datetime import datetime

import numpy as np
import pandas as pd

from Account import Account
from get_btc_info import get_btc_data
import strategy
import matplotlib.pyplot as plt
import matplotlib.dates as dates


class Backtest:
    def __init__(self, strategy_results, initial_capital, asset_parameters, stop_loss_threshold=0.05):
        self.account = Account(initial_capital)
        self.strategy_results = strategy_results
        self.asset_parameters = asset_parameters
        self.stop_loss_threshold = stop_loss_threshold
        self.open_positions = {}

    def check_liquidation(self, current_time, price_map):
        """
        在每个bar结束或价格更新后调用此方法，检查是否需要爆仓。
        :param current_time: 当前时刻（Datetime）
        :param price_map: dict, 当前各资产价格，例如 {'AAPL': 150.0, 'GOOG': 2800.0}
        """
        # 假设有一个维持保证金率，比如0.02（2%）
        maintenance_margin_rate = 0.02

        assets_to_liquidate = []

        for asset, pos in self.open_positions.items():
            quantity = pos['quantity']
            entry_price = pos['entry_price']
            borrowed_funds = pos['borrowed_funds']
            own_equity = pos['own_equity']
            hourly_rate = pos['hourly_rate']
            leverage_rate = pos['leverage_rate']
            position_type = pos['position_type']

            current_price = price_map.get(asset, None)
            if current_price is None:
                continue  # 无法获取当前价格则跳过

            # 计算当前持仓价值
            current_value = quantity * current_price

            # 计算持仓期间产生的利息
            holding_hours = (current_time - pos['entry_time']).total_seconds() / 3600
            interest = borrowed_funds * hourly_rate * holding_hours

            # 净值 = 当前持仓价值 + own_equity - borrowed_funds - interest - fees(若有)
            # 在此暂不考虑额外费用
            net_equity = current_value + own_equity - borrowed_funds - interest

            # 计算初始总头寸成本
            total_cost = own_equity + borrowed_funds

            # 爆仓条件示例：
            # 假定当净值/初始头寸成本 < 维持保证金率 时爆仓（实际中可能用更复杂的条件）
            margin_ratio = net_equity / total_cost if total_cost != 0 else 1

            if margin_ratio < maintenance_margin_rate:
                print(f"Liquidation triggered for {asset} at {current_time}! margin_ratio={margin_ratio:.4f}")
                assets_to_liquidate.append(asset)

        # 执行强制平仓
        for asset in assets_to_liquidate:
            current_price = price_map[asset]
            self.liquidate_position(asset, current_price, current_time)

    def liquidate_position(self, asset, price, current_time):
        """
        强制平仓，将仓位以当前价平掉，并在transaction中标记为liquidation交易。
        """
        pos = self.open_positions[asset]
        quantity = pos['quantity']
        borrowed_funds = pos['borrowed_funds']
        own_equity = pos['own_equity']
        holding_hours = (current_time - pos['entry_time']).total_seconds() / 3600
        interest = borrowed_funds * pos['hourly_rate'] * holding_hours
        fees = quantity * price * 0.001  # 0.1% 假设

        gross_revenue = quantity * price
        repay_amount = borrowed_funds + interest
        net_profit = gross_revenue - repay_amount - fees + own_equity

        # 先卖出资产获得USD
        self.account.sell(current_time, asset, quantity, "USD", gross_revenue,
                          own_equity=own_equity, borrowed_funds=borrowed_funds, interest=interest, fees=0)

        # 记录repay交易
        self.account.cash -= repay_amount
        repay_record = {
            'time': current_time,
            'type': 'repay',
            'description': 'Forced liquidation repay borrowed funds and interest',
            'borrowed_funds': borrowed_funds,
            'interest': interest,
            'amount': repay_amount
        }
        self.account.transaction.append(repay_record)

        # 支付费用
        self.account.cash -= fees
        fee_record = {
            'time': current_time,
            'type': 'fee',
            'description': 'Forced liquidation transaction fee',
            'fees': fees
        }
        self.account.transaction.append(fee_record)

        # 清除仓位信息
        self.account.clear_position_info(asset)
        del self.open_positions[asset]

        # 可以额外记录一次liquidation标记的交易或在sell中添加一个字段表示liquidation
        liquidation_record = {
            'time': current_time,
            'type': 'liquidation',
            'asset': asset,
            'price': price,
            'net_profit': net_profit
        }
        self.account.transaction.append(liquidation_record)

    def run(self):
        for daily_df in self.strategy_results:
            for idx, row in daily_df.iterrows():
                self.process_signal(row)
                self.check_stop_loss(row)
        return self.calculate_final_result()

    def process_signal(self, row):
        asset = row['asset']
        price = row['close']
        signal = row['signal']
        current_time = row['time']
        if signal == 1:  # 开多
            # 如果已经有同方向持仓，可以选择加仓逻辑，这里简单化不加仓，只在无持仓时才开仓
            if asset not in self.open_positions:
                self.open_long_position(asset, price, current_time)
        elif signal == -1:  # 开空
            if asset not in self.open_positions:
                self.open_short_position(asset, price, current_time)
        # signal = 0 不做操作

    def open_long_position(self, asset, price, current_time):
        min_unit = self.asset_parameters.get(asset, {}).get('min_trade_unit', 1)
        leverage = self.asset_parameters.get(asset, {}).get('leverage', 1)
        hourly_rate = self.asset_parameters.get(asset, {}).get('hourly_rate', 0.001)

        # 计算数量
        max_units = self.account.cash / price
        units_without_leverage = math.floor(max_units / min_unit) * min_unit
        if units_without_leverage <= 0:
            print("No cash to open long position.")
            return

        own_equity = units_without_leverage * price
        total_cost = own_equity * leverage
        borrowed_funds = total_cost - own_equity

        if self.account.cash < own_equity:
            print("Insufficient cash for own equity.")
            return

        own_equity_units = units_without_leverage
        borrowed_units = borrowed_funds / price
        quantity = own_equity_units + borrowed_units

        # 在开仓时记录保证金信息到account.margins中(可选)
        self.account.record_position_info(
            asset=asset,
            own_equity=own_equity,
            borrowed_funds=borrowed_funds,
            leverage_rate=leverage,
            hourly_rate=hourly_rate
        )

        # 第一步：用own_equity购买资产
        # 此时我们可将own_equity等信息直接记录到交易中，但利息和费用此时还没有产生或计算，为0或None即可
        self.account.buy(current_time, asset, own_equity_units, "USD", own_equity,
                         own_equity=own_equity, borrowed_funds=0, interest=0, fees=0)

        # 第二步：借入剩余的资产
        if borrowed_units > 0:
            # 借入部分也可标记own_equity=0，borrowed_funds不为0表示是借入的
            self.account.buy(current_time, asset, borrowed_units, "USD", 0,
                             own_equity=0, borrowed_funds=borrowed_funds, interest=0, fees=0)

        self.open_positions[asset] = {
            'entry_time': current_time,
            'entry_price': price,
            'quantity': quantity,
            'own_equity': own_equity,
            'borrowed_funds': borrowed_funds,
            'leverage_rate': leverage,
            'hourly_rate': hourly_rate,
            'position_type': 'long'
        }

    def open_short_position(self, asset, price, current_time):
        min_unit = self.asset_parameters.get(asset, {}).get('min_trade_unit', 1)
        leverage = self.asset_parameters.get(asset, {}).get('leverage', 1)
        hourly_rate = self.asset_parameters.get(asset, {}).get('hourly_rate', 0.001)

        max_units = self.account.cash / price
        units_without_leverage = math.floor(max_units / min_unit) * min_unit
        if units_without_leverage <= 0:
            print("No cash margin to open short position.")
            return

        own_equity = units_without_leverage * price
        total_cost = own_equity * leverage
        borrowed_funds = total_cost - own_equity

        if self.account.cash < own_equity:
            print("Insufficient cash for short margin.")
            return

        # 扣除自有资金作为保证金
        self.account.cash -= own_equity

        # 借入总计相当于total_cost/price的资产数量
        borrowed_units = total_cost / price
        # 第一步：借入资产(ask_asset_amount=0)
        self.account.buy(current_time, asset, borrowed_units, asset, 0)
        # 第二步：卖出借来的全部资产，获得 total_cost USD
        self.account.sell(current_time, asset, borrowed_units, "USD", total_cost)

        # 此时账户中增加total_cost USD，但我们减少了own_equity USD的保证金，所以净现金增加borrowed_funds。
        # 仓位记录
        self.open_positions[asset] = {
            'entry_time': current_time,
            'entry_price': price,
            'quantity': borrowed_units,
            'own_equity': own_equity,
            'borrowed_funds': borrowed_funds,
            'leverage_rate': leverage,
            'hourly_rate': hourly_rate,
            'position_type': 'short'
        }

    def close_position(self, asset, price, current_time):
        pos = self.open_positions[asset]
        quantity = pos['quantity']
        borrowed_funds = pos['borrowed_funds']
        own_equity = pos['own_equity']
        holding_hours = (current_time - pos['entry_time']).total_seconds() / 3600
        interest = borrowed_funds * pos['hourly_rate'] * holding_hours
        fees = quantity * price * 0.001  # 假设0.1%手续费

        # 卖出所持资产
        gross_revenue = quantity * price
        # 第一步：执行卖出操作，将资产换成USD
        self.account.sell(
            timestamp=current_time,
            sell_asset_type=asset,
            sell_asset_amount=quantity,
            receive_asset_type="USD",
            receive_asset_amount=gross_revenue,
            own_equity=own_equity,
            borrowed_funds=borrowed_funds,
            interest=interest,
            fees=0  # 此处fees暂不计入，因为还没正式扣除
        )

        # 此时 account.cash 增加了 gross_revenue
        # 第二步：偿还借款和利息，从account.cash中扣除repay_amount
        repay_amount = borrowed_funds + interest
        if self.account.cash < repay_amount:
            print("Warning: Not enough cash to repay borrowed funds and interest!")
        self.account.cash -= repay_amount
        # 记录一条repay交易，让交易记录清晰
        repay_record = {
            'time': current_time,
            'type': 'repay',
            'description': 'Repay borrowed funds and interest',
            'borrowed_funds': borrowed_funds,
            'interest': interest,
            'amount': repay_amount
        }
        self.account.transaction.append(repay_record)

        # 第三步：支付交易费用
        if self.account.cash < fees:
            print("Warning: Not enough cash to pay fees!")
        self.account.cash -= fees
        fee_record = {
            'time': current_time,
            'type': 'fee',
            'description': 'Transaction fees',
            'fees': fees
        }
        self.account.transaction.append(fee_record)

        # 最后：own_equity是我们原本投入的本金，net_profit = gross_revenue - repay_amount - fees + own_equity
        # 实际上，这些计算你已经做完了，不过在这里分步做的话，
        # net_profit实际上已经通过这些增减操作体现在account.cash中。

        # 清除保证金信息和仓位信息
        self.account.clear_position_info(asset)
        del self.open_positions[asset]

    def check_stop_loss(self, row):
        current_time = row['time']
        price = row['close']
        assets_to_close = []
        for a, pos in self.open_positions.items():
            entry_price = pos['entry_price']
            if pos['position_type'] == 'long':
                # 多头止损条件
                if price < entry_price * (1 - self.stop_loss_threshold):
                    assets_to_close.append(a)
            else:
                # 空头止损条件（价格上涨超出阈值）
                if price > entry_price * (1 + self.stop_loss_threshold):
                    assets_to_close.append(a)

        for a in assets_to_close:
            self.close_position(a, price, current_time)

    def calculate_final_result(self):
        final_cash = self.account.cash
        return {
            'final_cash': final_cash,
            'transaction_history': self.account.get_transaction_history()
        }