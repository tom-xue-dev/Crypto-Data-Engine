from Account import Account

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
