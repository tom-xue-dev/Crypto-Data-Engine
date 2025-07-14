from abc import ABC, abstractmethod


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
