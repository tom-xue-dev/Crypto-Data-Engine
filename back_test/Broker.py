from Account import Account

class Broker:
    """
    Broker 为撮合类，对回测交易的函数调用高层封装
    """

    def __init__(self, account: Account, leverage_manager=None, stop_loss_logic=None,fees = 0.001):
        """
        撮合订单的类
        传入账号据数据用于记录交易历史
        传入市场类型用于记录现货或者期货市场
        传入leverage_manager用于记录每小时末的杠杆费率结算
        传入stop_loss_logic用于进行止损的逻辑处理

        :param account:
        :param leverage_manager:
        :param stop_loss_logic:
        :param fees: 手续费，默认千1
        """
        self.fees = fees
        self.account = account
        self.leverage_manager = leverage_manager
        self.stop_loss_logic = stop_loss_logic

    def open_position(self, asset, direction, quantity, price, leverage, position_type, current_time):
        cost = quantity * price / leverage
        # 计算手续费 现在默认是maker: 双向千1
        cost = cost * (1+self.fees)
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

        if (asset, direction) not in self.account.positions:
            self.account.positions[(asset, direction)] = pos
            if direction == "short": #增加冻结保证金
                self.account.reversed_cash += quantity * price / leverage
        else:
            existing_pos = self.account.positions[(asset, direction)]
            old_quantity = existing_pos.quantity
            new_quantity = old_quantity + quantity
            weighted_price = (old_quantity * existing_pos.entry_price + quantity * price) / new_quantity
            existing_pos.quantity = new_quantity
            existing_pos.entry_price = weighted_price
            self.account.positions[(asset, direction)] = existing_pos


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
            self.stop_loss_logic.init_holding(asset=asset, price=self.account.positions[(asset, direction)].entry_price,
                                              direction=direction)

    def close_position(self, asset, direction, price, current_time, stop_loss=False):
        key = (asset, direction)
        if key not in self.account.positions:
            raise ValueError(f"target asset{key} not in holdings")
        pos = self.account.positions.pop(key)

        if self.stop_loss_logic:
            # print(f"close{asset, direction} at {current_time}",pos.entry_price,price)
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
            self.account.reversed_cash -= pos.quantity * pos.entry_price #释放保证金

        total_gain = (pos.entry_price * pos.quantity + pnl)
        self.account.cash += total_gain * (1 - self.fees) # 千分之一手续费
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

    def clear_position(self,price_map):
        for (asset, direction), pos in self.account.positions.items():
            exit_price = price_map.get(asset)
            if exit_price is None:
                print(f"warning,cannot find price_map{asset, price_map.get(asset)}")
                continue
            if direction == "long":
                pnl = pos.quantity * (exit_price - pos.entry_price)
            else:
                pnl = -pos.quantity * (exit_price - pos.entry_price)

            total_gain = (pos.entry_price * pos.quantity + pnl)
            self.account.cash += total_gain * (1 - self.fees)  # 千分之一手续费
            self.account.record_transaction({
                "time": "exit",
                "action": "close",
                "asset": asset,
                "direction": direction,
                "quantity": pos.quantity,
                "close_price": exit_price,
                "pnl": pnl,
                "stop_loss": "exit"
            })

    def on_bar_end(self, current_time, price_map, asset,**kwargs):
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
            atr_value = kwargs.get('atr_value')
            positions_to_close = self.stop_loss_logic.check_stop_loss(account=self.account, price_map=price_map,atr_value = atr_value,
                                                                      current_time=current_time, update_asset = asset,holding=1)

            for (asset, direction) in positions_to_close:
                if price_map.get(asset) is not None:
                    if (asset, direction) not in self.account.positions:
                        raise ValueError(f"{asset},{direction} not found")
                    self.close_position(asset, direction, price_map.get(asset), current_time, stop_loss=True)
                else:
                    print(f"warning,cannot find price_map{asset, price_map.get(asset)}")

