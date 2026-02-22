import math
from datetime import datetime

from crypto_data_engine.services.back_test.portfolio import Portfolio, OrderExecutor
from crypto_data_engine.services.back_test.config import CostConfigModel
from crypto_data_engine.core.interfaces import OrderSide
from crypto_data_engine.core.base import TradeDirection


def test_open_close_short_and_slippage_signs():
    portfolio = Portfolio(initial_capital=10_000)
    cost = CostConfigModel(commission_rate=0.0, taker_rate=0.0, maker_rate=0.0, slippage_rate=0.001)
    ex = OrderExecutor(portfolio=portfolio, cost_config=cost)

    base_price = 100.0

    # Open short should use SELL side internally and succeed
    order_open = ex.open_short("AAA", quantity=10, price=base_price, leverage=1.0)
    assert order_open is not None
    assert order_open.status.name == "FILLED"
    # SELL slippage: filled price should be lower than base
    assert order_open.filled_price < base_price

    # Close short should use BUY side and succeed
    order_close = ex.close_short("AAA", quantity=10, price=base_price)
    assert order_close is not None
    assert order_close.status.name == "FILLED"
    # BUY slippage: filled price should be higher than base
    assert order_close.filled_price > base_price


def test_fill_price_slippage_direction_long():
    portfolio = Portfolio(initial_capital=1_000)
    cost = CostConfigModel(commission_rate=0.0, taker_rate=0.0, maker_rate=0.0, slippage_rate=0.002)
    ex = OrderExecutor(portfolio=portfolio, cost_config=cost)

    price = 50.0
    # BUY -> price up by slippage
    buy_order = ex.execute_market_order("BBB", OrderSide.BUY, quantity=1, price=price, direction=TradeDirection.LONG)
    assert buy_order is not None
    assert buy_order.filled_price > price

    # SELL closing long -> price down by slippage
    sell_order = ex.execute_market_order("BBB", OrderSide.SELL, quantity=1, price=price, direction=TradeDirection.LONG)
    assert sell_order is not None
    assert sell_order.filled_price < price

