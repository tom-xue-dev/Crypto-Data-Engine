class Account:
    def __init__(self, initial_cash, asset_type=""):
        """
        初始化账户，设置初始现金、资产类别和资产数量
        :param initial_cash: float, 初始金额
        :param asset_type: str, 金融资产类别 (例如 "Bitcoin", "Stock", "Gold" 等)
        """
        self.cash = initial_cash  # 当前现金
        self.asset_type = asset_type  # 金融资产类别
        self.assets = 0.0  # 当前资产数量

    def set_asset_type(self, asset_type):
        """
        设置金融资产类别
        :param asset_type: str, 新的金融资产类别
        """
        self.asset_type = asset_type
        print(f"Asset type set to '{self.asset_type}'.")

    def current_status(self):
        """
        返回当前账户状态
        """
        return (f"Current Cash: ${self.cash:.2f}, "
                f"{self.asset_type}: {self.assets:.6f} units")

    def buy(self, amount_in_usd, asset_price):
        """
        买入金融资产
        :param amount_in_usd: float, 买入金额 (美元)
        :param asset_price: float, 当前资产价格 (美元/单位)
        """
        if amount_in_usd > self.cash:
            print("Insufficient funds to buy.")
            return

        units_to_buy = amount_in_usd / asset_price
        self.cash -= amount_in_usd
        self.assets += units_to_buy
        print(f"Bought {units_to_buy:.6f} "
              f"{self.asset_type} for ${amount_in_usd:.2f}.")

    def sell(self, amount_in_units, asset_price):
        """
        卖出金融资产
        :param amount_in_units: float, 卖出资产数量
        :param asset_price: float, 当前资产价格 (美元/单位)
        """
        if amount_in_units > self.assets:
            print("Insufficient assets to sell.")
            return

        usd_earned = amount_in_units * asset_price
        self.assets -= amount_in_units
        self.cash += usd_earned
        print(f"Sold {amount_in_units:.6f} "
              f"{self.asset_type} for ${usd_earned:.2f}.")
