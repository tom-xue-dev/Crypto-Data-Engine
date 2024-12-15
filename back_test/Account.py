import datetime
from collections import defaultdict
import pandas as pd


class Account:
    def __init__(self, initial_cash: float):
        """
        初始化账户，设置初始现金、持仓以及保证金信息结构。
        """
        self.cash = initial_cash  # 当前现金（USD）
        self.spot_holdings = {}  # 现货持仓: { 'BTC': {'amount': 5, 'leverage': 2x}, ... }
        self.futures_positions = {}  # 期货持仓: { 'BTCUSDT_PERP': {'side': 'LONG','qty':...,'entry_price':..., 'leverage':...}, ... }
        self.transactions = []  # 交易记录列表
        self.margin_balance = initial_cash  # 合约保证金余额(USDT), 可根据策略从self.cash划转

    def buy(self, timestamp: datetime.datetime, bid_asset_type: str, bid_asset_amount: float, ask_asset_type: str,
            ask_asset_amount: float, own_equity: float = None, borrowed_funds: float = None, interest: float = 0.0, fees: float = 0.0):
        """
        扩展buy方法：增加可选的own_equity、borrowed_funds、interest、fees字段，以记录此笔交易的底层资金情况。

        own_equity: 本金投入
        borrowed_funds: 借入资金
        interest: 此笔交易涉及的利息
        fees: 手续费

        这些字段在真实逻辑中由Backtest层计算完成并传入此处，仅做记录。
        """
        if ask_asset_amount > 0:
            if ask_asset_type == "USD":
                if self.cash < ask_asset_amount:
                    raise ValueError("现金不足，无法完成交易。")
                self.cash -= ask_asset_amount
            else:
                if ask_asset_type not in self.holdings or self.holdings[ask_asset_type] < ask_asset_amount:
                    raise ValueError(f"持仓中没有足够的 {ask_asset_type} 来支付此交易。交易失败")
                self.holdings[ask_asset_type] -= ask_asset_amount
                if self.holdings[ask_asset_type] == 0:
                    del self.holdings[ask_asset_type]

        # 增加所买入的资产
        if bid_asset_type not in self.holdings:
            self.holdings[bid_asset_type] = 0
        self.holdings[bid_asset_type] += bid_asset_amount

        # 记录交易，增加新字段
        transaction_record = {
            'time': timestamp,
            'type': 'buy',
            'receive_asset_type': bid_asset_type,
            'receive_asset_amount': bid_asset_amount,
            'cost_asset_type': ask_asset_type,
            'cost_asset_amount': ask_asset_amount,
            'own_equity': own_equity,
            'borrowed_funds': borrowed_funds,
            'interest': interest,
            'fees': fees
        }
        self.transaction.append(transaction_record)

    def sell(self, timestamp: datetime.datetime, sell_asset_type: str, sell_asset_amount: float,
             receive_asset_type: str, receive_asset_amount: float,
             own_equity: float = None, borrowed_funds: float = None,
             interest: float = 0.0, fees: float = 0.0):
        """
        扩展sell方法，增加own_equity、borrowed_funds、interest、fees字段。
        backtest层会根据实际情况计算好这些参数传入Account。

        :param timestamp:
        :param sell_asset_type: 出售的资产类型 (如 "AAPL")
        :param sell_asset_amount: 出售的数量
        :param receive_asset_type: 收到的资产类型 (如 "USD")
        :param receive_asset_amount: 收到的数量
        :param own_equity: 本金部分（如果此笔交易涉及平仓，own_equity可以表示归还本金的金额）
        :param borrowed_funds: 借入资金部分（若本次平仓还需要归还借款）
        :param interest: 本笔交易发生的利息费用（若有）
        :param fees: 手续费
        """
        if sell_asset_type not in self.holdings or self.holdings[sell_asset_type] < sell_asset_amount:
            raise ValueError(f"持仓中没有足够的 {sell_asset_type} 来完成出售。交易失败")

        self.holdings[sell_asset_type] -= sell_asset_amount
        if self.holdings[sell_asset_type] == 0:
            del self.holdings[sell_asset_type]

        if receive_asset_type == "USD":
            self.cash += receive_asset_amount
        else:
            if receive_asset_type not in self.holdings:
                self.holdings[receive_asset_type] = 0
            self.holdings[receive_asset_type] += receive_asset_amount

        # 记录交易
        transaction_record = {
            'time': timestamp,
            'type': 'sell',
            'sell_asset_type': sell_asset_type,
            'sell_asset_amount': sell_asset_amount,
            'receive_asset_type': receive_asset_type,
            'receive_asset_amount': receive_asset_amount,
            'own_equity': own_equity,
            'borrowed_funds': borrowed_funds,
            'interest': interest,
            'fees': fees
        }

        self.transaction.append(transaction_record)

    def get_transaction_history(self):
        return self.transaction

    def calculate_total_value(self, price_map: dict):
        """
        根据给定的资产价格字典计算当前账户总价值
        """
        total_value = self.cash
        for asset, qty in self.holdings.items():
            price = price_map.get(asset, 0)
            total_value += qty * price
        return total_value

    def current_status(self):
        """
        返回当前账户状态，包括现金和持仓信息。
        如果需要显示保证金信息，也可在此打印 self.margins。
        """
        return f"Current Cash: ${self.cash:.2f}, Holdings: {self.holdings}, Margins: {self.margins}"

    def record_position_info(self, asset: str, own_equity: float, borrowed_funds: float, leverage_rate: float,
                             hourly_rate: float):
        """
        在建立或更新杠杆仓位时，记录或更新保证金相关信息。
        由Backtest在开仓或调整仓位时调用。
        """
        self.margins[asset] = {
            'own_equity': own_equity,
            'borrowed_funds': borrowed_funds,
            'leverage_rate': leverage_rate,
            'hourly_rate': hourly_rate
        }

    def clear_position_info(self, asset: str):
        """
        清除对某资产的保证金记录（平仓后调用）。
        """
        if asset in self.margins:
            del self.margins[asset]
