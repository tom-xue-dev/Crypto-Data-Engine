from datetime import datetime

from crypto_data_engine.services.back_test.portfolio import Portfolio
from crypto_data_engine.core.base import TradeDirection


def test_nav_short_with_leverage_marks_to_equity():
    ts = datetime(2024, 1, 1)
    p = Portfolio(initial_capital=1000.0)

    # Open short: 1 unit at 100, leverage 10: margin=10 deducted from cash
    p.open_position("AAA", TradeDirection.SHORT, quantity=1.0, price=100.0, timestamp=ts, leverage=10.0, fees=0.0)
    # After open at same price, NAV should remain close to 1000 (allow minor fp tolerance)
    p.update_prices({"AAA": 100.0})
    assert abs(p.nav - 1000.0) < 1e-6

    # If price goes down to 90, unrealized profit = 10; NAV should increase
    p.update_prices({"AAA": 90.0})
    assert abs(p.nav - 1010.0) < 1e-6

    # If price goes up to 110, unrealized loss = -10; NAV should decrease
    p.update_prices({"AAA": 110.0})
    assert abs(p.nav - 990.0) < 1e-6
