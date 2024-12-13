import datetime
from collections import defaultdict

import pandas as pd


class Account:
    def __init__(self, initial_cash: int, asset_type: list):
        """
        初始化账户，设置初始现金、资产类别和资产数量
        :param initial_cash: float, 初始金额
        :param asset_type: list, 金融资产类别 (例如 "Bitcoin", "Stock", "Gold" 等)

        cash : 当前持有现金，均为USD
        holdings: 当前持有的金融资产和数量，例如{'BTC':10} 代表持有10个BTC
        transaction: 记录所有成交记录，每条记录可以包含交易类型、时间、资产类型、数量等信息。
        """
        self.cash = initial_cash  # 当前现金
        self.holdings = {}  # 资产持仓字典
        self.transaction = []  # 交易记录列表

    def current_status(self):
        """
        返回当前账户状态
        """
        return f"Current Cash: ${self.cash:.2f}, Holdings: {self.holdings}"

    def buy(self, timestamp: datetime.datetime, bid_asset_type: str, bid_asset_amount: float, ask_asset_type: str,
            ask_asset_amount: float):
        """
        从账户中支付 ask_asset_type 来获得 bid_asset_type

        :param timestamp: 买入的具体时间
        :param bid_asset_type: 要买入的资产类型 (如 "BTC")
        :param bid_asset_amount: 要买入的数量 (如 2.0)
        :param ask_asset_type: 支付的资产类型 ("USD" 或其他已持有的资产类型)
        :param ask_asset_amount: 支付的数量 (如 3000 USD 或者 10个ETH)
        :return: None
        """

        # 检查支付资产是否为USD，如果是则从现金中支付
        if ask_asset_type == "USD":
            if self.cash < ask_asset_amount:
                raise ValueError("现金不足，无法完成交易。")
            self.cash -= ask_asset_amount
        else:
            # 非USD资产付款时，从 holdings 扣除对应数量资产
            if ask_asset_type not in self.holdings or self.holdings[ask_asset_type] < ask_asset_amount:
                raise ValueError(f"持仓中没有足够的 {ask_asset_type} 来支付此交易。交易失败")
            self.holdings[ask_asset_type] -= ask_asset_amount
            # 如果某资产数量减为0，可以根据需求选择是否删除该键
            if self.holdings[ask_asset_type] == 0:
                del self.holdings[ask_asset_type]

        # 增加所买入的资产
        if bid_asset_type not in self.holdings:
            self.holdings[bid_asset_type] = 0
        self.holdings[bid_asset_type] += bid_asset_amount

        # 记录交易
        transaction_record = {
            'time': timestamp,
            'type': 'buy',
            'bid_asset_type': bid_asset_type,
            'bid_asset_amount': bid_asset_amount,
            'ask_asset_type': ask_asset_type,
            'ask_asset_amount': ask_asset_amount
        }
        self.transaction.append(transaction_record)

    def sell(self, timestamp: datetime.datetime, sell_asset_type: str, sell_asset_amount: float,
             receive_asset_type: str, receive_asset_amount: float):
        """
        出售 sell_asset_type 来获得 receive_asset_type
        :param timestamp:
        :param sell_asset_type: 要出售的资产类型 (如 "BTC")
        :param sell_asset_amount: 要出售的数量 (如 2.0)
        :param receive_asset_type: 收到的资产类型 (如 "USD" 或其他资产类型)
        :param receive_asset_amount: 收到的数量 (如 3000 USD 或 10个ETH)
        :return: None
        """
        # 检查持有的资产是否足够出售
        if sell_asset_type not in self.holdings or self.holdings[sell_asset_type] < sell_asset_amount:
            raise ValueError(f"持仓中没有足够的 {sell_asset_type} 来完成出售。交易失败")

        # 扣除出售的资产
        self.holdings[sell_asset_type] -= sell_asset_amount

        # 如果该资产数量为0，可以选择删除该键
        if self.holdings[sell_asset_type] == 0:
            del self.holdings[sell_asset_type]

        # 增加收到的资产
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
            'receive_asset_amount': receive_asset_amount
        }
        self.transaction.append(transaction_record)

    def get_transaction_history(self):
        return self.transaction

    def calculate_daily_nav_changes(self):
        """计算每日净值变化"""
        # 创建一个字典来记录每个日期的净值
        daily_nav = defaultdict(float)

        # 遍历每笔交易，记录每日资产变动
        for transaction in self.transaction:
            date = transaction['time'].date()

            # 计算交易对净值的影响
            if transaction['type'] == 'sell':
                daily_nav[date] += transaction['receive_asset_amount']
                daily_nav[date] -= transaction['sell_asset_amount']
            elif transaction['type'] == 'buy':
                daily_nav[date] -= transaction['ask_asset_amount']
                daily_nav[date] += transaction['bid_asset_amount']

        # 将字典转换为DataFrame，按日期排序
        nav_df = pd.DataFrame(list(daily_nav.items()), columns=['Date', 'NAV'])
        nav_df.sort_values(by='Date', inplace=True)

        # 计算每日净值变化
        nav_df['Daily Change'] = nav_df['NAV'].diff().fillna(0)

        return nav_df
