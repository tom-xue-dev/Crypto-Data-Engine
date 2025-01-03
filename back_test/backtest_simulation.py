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

    def __init__(self, account: Account, leverage_manager=None, stop_loss_logic=DefaultStopLossLogic()):
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
        # print(f"open pos in {current_time}")
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
        if self.stop_loss_logic:
            self.stop_loss_logic.highest_price_map[asset] = price
            self.stop_loss_logic.lowest_price_map[asset] = price

    def close_position(self, asset, direction, price, current_time, stop_loss=False):
        key = (asset, direction)
        if key not in self.account.positions:
            raise ValueError("target asset not in holdings")

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
            "pnl": pnl,
            "stop_loss": stop_loss
        })

        if self.stop_loss_logic:
            del self.stop_loss_logic.highest_price_map[asset]
            del self.stop_loss_logic.lowest_price_map[asset]

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
                                                                      current_time=current_time)
            for (asset, direction) in positions_to_close:
                self.close_position(asset, direction, price_map.get(asset), current_time, stop_loss=True)


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
        self.net_value_history = []

    def run(self):
        for df in self.strategy_results:
            current_market_cap = self.get_market_cap(df)
            price_map = {}
            current_time = df['time'][0]
            for idx, row in df.iterrows():
                asset = row['asset']
                price = row['close']
                signal = row['signal']

                # 1) 处理交易信号
                self.process_signal(signal, asset, price, current_time, current_market_cap)

                price_map[asset] = price
            # 2) bar结束调用 on_bar_end
            self.broker.on_bar_end(current_time, price_map)
            # 3) 记录当前净值
            self.log_net_value(df, current_time)

        return {
            "final_cash": self.broker.account.cash,
            "positions": self.broker.account.positions,
            "transactions": self.broker.account.transactions,
            "net_value_history": pd.DataFrame(self.net_value_history)
        }

    def process_signal(self, signal, asset, price, current_time, current_market_cap):
        # 简化: signal=1 -> 开多, signal=-1 -> 开空, signal=0 -> 不操作
        position = self.pos_manager.get_allocate_pos(current_market_cap, self.broker.account.cash)
        quantity = position / price
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
            if existing_short_key in holdings:
                self.broker.close_position(asset, "short", price, current_time)
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
            if existing_long_key in holdings:
                self.broker.close_position(asset, "long", price, current_time)
            if existing_short_key in holdings:
                return
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
            "net_value": net_value,
            "cash": self.broker.account.cash,
            "positions": list(self.broker.account.positions.keys())
        })


class PerformanceAnalyzer:
    def __init__(self, net_value_df: pd.DataFrame):
        """
        net_value_df: DataFrame，至少包含 ['time', 'net_value'] 字段。
        时间可以是索引或者一列，根据你的数据格式定。
        """
        self.net_value_df = net_value_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """
        将 time 转换为 datetime，并按时间排序等预处理。
        也可以在这里计算日度收益率、累计收益等。
        """
        # 如果 time 不是 datetime，需要先转换
        if not np.issubdtype(self.net_value_df['time'].dtype, np.datetime64):
            self.net_value_df['time'] = pd.to_datetime(self.net_value_df['time'])

        # 按时间排序
        self.net_value_df.sort_values(by='time', inplace=True)

        # 设置 time 为索引（可选）
        self.net_value_df.set_index('time', inplace=True)

        # 计算收益率 (如按 bar 计算)
        self.net_value_df['returns'] = self.net_value_df['net_value'].pct_change().fillna(0)

    def plot_net_value(self):
        """
        绘制净值曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.net_value_df.index, self.net_value_df['net_value'], label='Net Value')
        plt.title('Net Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Net Value')
        plt.legend()
        plt.show()

    def calculate_max_drawdown(self) -> float:
        """
        计算最大回撤
        """
        cum_max = self.net_value_df['net_value'].cummax()
        drawdown = self.net_value_df['net_value'] / cum_max - 1
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_annual_return(self, annual_factor: int = 252) -> float:
        """
        计算年化收益率（假设每年有 annual_factor 个交易日/交易bar，可根据实际情况调整）
        """
        # 先计算累计收益率
        final_net_value = self.net_value_df['net_value'].iloc[-1]
        init_net_value = self.net_value_df['net_value'].iloc[0]
        total_return = final_net_value / init_net_value - 1

        # 计算回测总天数
        total_days = (self.net_value_df.index[-1] - self.net_value_df.index[0]).days
        if total_days == 0:
            return 0.0

        # 年化系数（粗略）
        yearly_periods = total_days / 365
        annual_return = (1 + total_return) ** (1 / yearly_periods) - 1
        return annual_return

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0, annual_factor: int = 252) -> float:
        """
        计算夏普比率:
        Sharpe = (Mean(returns) - risk_free_rate) / Std(returns) * sqrt(annual_factor)
        """
        rets = self.net_value_df['returns']
        mean_ret = rets.mean()
        std_ret = rets.std()
        if std_ret == 0:
            return 0.0
        sharpe = (mean_ret - risk_free_rate / annual_factor) / std_ret * np.sqrt(annual_factor)
        return sharpe

    def summary(self):
        """
        输出常见指标的汇总信息
        """
        max_dd = self.calculate_max_drawdown()
        ann_ret = self.calculate_annual_return()
        sharpe = self.calculate_sharpe_ratio()

        print("Performance Summary:")
        print(f"Final Net Value: {self.net_value_df['net_value'].iloc[-1]:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Annual Return: {ann_ret:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
