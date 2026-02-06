"""
Unit tests for Portfolio, Position, and OrderExecutor.
"""
from datetime import datetime, timedelta

import pytest

from crypto_data_engine.core.base import TradeDirection
from crypto_data_engine.core.interfaces import OrderSide, OrderStatus
from crypto_data_engine.services.back_test.config import CostConfigModel
from crypto_data_engine.services.back_test.portfolio import (
    Position,
    Portfolio,
    OrderExecutor,
)


# =============================================================================
# Position Tests
# =============================================================================

class TestPosition:
    """Tests for Position class."""

    def test_create_position(self):
        """Test creating a new position."""
        pos = Position(
            asset="BTCUSDT",
            direction=TradeDirection.LONG,
        )
        
        assert pos.asset == "BTCUSDT"
        assert pos.direction == TradeDirection.LONG
        assert pos.quantity == 0.0
        assert not pos.is_open

    def test_create_long_factory(self):
        """Test factory method for long position."""
        timestamp = datetime.now()
        pos = Position.create_long(
            asset="BTCUSDT",
            quantity=1.0,
            price=50000,
            timestamp=timestamp,
        )
        
        assert pos.asset == "BTCUSDT"
        assert pos.direction == TradeDirection.LONG
        assert pos.quantity == 1.0
        assert pos.entry_price == 50000
        assert pos.is_open

    def test_create_short_factory(self):
        """Test factory method for short position."""
        timestamp = datetime.now()
        pos = Position.create_short(
            asset="ETHUSDT",
            quantity=10.0,
            price=3000,
            timestamp=timestamp,
        )
        
        assert pos.direction == TradeDirection.SHORT
        assert pos.quantity == 10.0
        assert pos.is_open

    def test_market_value(self):
        """Test market value calculation."""
        pos = Position.create_long("BTC", 2.0, 50000, datetime.now())
        pos.update_price(55000)
        
        assert pos.market_value == 110000  # 2 * 55000

    def test_unrealized_pnl_long(self):
        """Test unrealized PnL for long position."""
        pos = Position.create_long("BTC", 1.0, 50000, datetime.now())
        pos.update_price(55000)
        
        assert pos.unrealized_pnl == 5000  # Gain of 5000

    def test_unrealized_pnl_short(self):
        """Test unrealized PnL for short position."""
        pos = Position.create_short("BTC", 1.0, 50000, datetime.now())
        pos.update_price(45000)
        
        # Short profits when price goes down
        assert pos.unrealized_pnl == 5000

    def test_add_quantity(self):
        """Test adding to position."""
        timestamp = datetime.now()
        pos = Position.create_long("BTC", 1.0, 50000, timestamp)
        pos.add_quantity(1.0, 52000, timestamp + timedelta(hours=1))
        
        assert pos.quantity == 2.0
        assert pos.entry_price == 51000  # Average: (50000 + 52000) / 2

    def test_reduce_quantity(self):
        """Test reducing position."""
        timestamp = datetime.now()
        pos = Position.create_long("BTC", 2.0, 50000, timestamp)
        pos.update_price(55000)
        
        pnl = pos.reduce_quantity(1.0, 55000, timestamp + timedelta(hours=1))
        
        assert pos.quantity == 1.0
        assert pnl == 5000  # Realized PnL from 1 BTC
        assert pos.realized_pnl == 5000

    def test_close_position(self):
        """Test closing entire position."""
        timestamp = datetime.now()
        pos = Position.create_long("BTC", 1.0, 50000, timestamp)
        
        pnl = pos.close(55000, timestamp + timedelta(hours=1))
        
        assert not pos.is_open
        assert pnl == 5000
        assert len(pos.trades) == 1

    def test_high_water_mark_long(self):
        """Test high water mark tracking for long."""
        pos = Position.create_long("BTC", 1.0, 50000, datetime.now())
        
        pos.update_price(55000)
        assert pos.high_water_mark == 55000
        
        pos.update_price(53000)
        assert pos.high_water_mark == 55000  # Should not decrease
        
        pos.update_price(60000)
        assert pos.high_water_mark == 60000

    def test_drawdown(self):
        """Test drawdown calculation."""
        pos = Position.create_long("BTC", 1.0, 50000, datetime.now())
        pos.update_price(55000)  # New high
        pos.update_price(49500)  # Drop
        
        # Drawdown = (55000 - 49500) / 55000 = 10%
        assert abs(pos.drawdown - 0.1) < 0.001

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now()
        pos = Position.create_long("BTC", 1.0, 50000, timestamp)
        pos.update_price(55000)
        
        data = pos.to_dict()
        
        assert data["asset"] == "BTC"
        assert data["direction"] == "long"
        assert data["quantity"] == 1.0
        assert data["market_value"] == 55000
        assert data["unrealized_pnl"] == 5000


# =============================================================================
# Portfolio Tests
# =============================================================================

class TestPortfolio:
    """Tests for Portfolio class."""

    def test_create_portfolio(self):
        """Test creating a new portfolio."""
        portfolio = Portfolio(initial_capital=1_000_000)
        
        assert portfolio.initial_capital == 1_000_000
        assert portfolio.cash == 1_000_000
        assert portfolio.nav == 1_000_000

    def test_open_long_position(self):
        """Test opening a long position."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        pos = portfolio.open_position(
            asset="BTCUSDT",
            direction=TradeDirection.LONG,
            quantity=1.0,
            price=50000,
            timestamp=timestamp,
        )
        
        assert pos.is_open
        assert pos.quantity == 1.0
        assert portfolio.cash == 50000  # 100000 - 50000

    def test_open_short_position(self):
        """Test opening a short position."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        portfolio.open_position(
            asset="BTCUSDT",
            direction=TradeDirection.SHORT,
            quantity=1.0,
            price=50000,
            timestamp=timestamp,
        )
        
        assert portfolio.short_exposure == 50000

    def test_close_position_with_profit(self):
        """Test closing position with profit."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        portfolio.update_prices({"BTC": 55000})
        
        pnl = portfolio.close_position("BTC", TradeDirection.LONG, None, 55000, timestamp)
        
        assert pnl == 5000
        assert portfolio.cash == 105000  # Initial 100k + 5k profit

    def test_close_position_with_loss(self):
        """Test closing position with loss."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        
        pnl = portfolio.close_position("BTC", TradeDirection.LONG, None, 45000, timestamp)
        
        assert pnl == -5000
        assert portfolio.cash == 95000

    def test_nav_calculation(self):
        """Test NAV calculation with open position."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        portfolio.update_prices({"BTC": 60000})
        
        # NAV = Cash (50000) + Position Value (60000) = 110000
        assert portfolio.nav == 110000

    def test_get_weights(self):
        """Test getting portfolio weights."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        portfolio.update_prices({"BTC": 50000})
        
        weights = portfolio.get_weights()
        
        # Weight = Position Value / NAV = 50000 / 100000 = 0.5
        assert weights["BTC"] == 0.5

    def test_leverage_calculation(self):
        """Test leverage calculation."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        # Open positions worth 150% of NAV using leverage
        portfolio.open_position(
            "BTC", TradeDirection.LONG, 2.0, 50000, timestamp,
            leverage=2.0,  # Using 2x leverage
        )
        portfolio.update_prices({"BTC": 50000})
        
        # Gross exposure = 100000
        # NAV should be around 100000 (cash used = 50000, position value = 100000)
        assert portfolio.gross_exposure == 100000

    def test_exposure_metrics(self):
        """Test exposure calculations."""
        portfolio = Portfolio(initial_capital=200_000)
        timestamp = datetime.now()
        
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        portfolio.open_position("ETH", TradeDirection.SHORT, 10.0, 3000, timestamp)
        portfolio.update_prices({"BTC": 50000, "ETH": 3000})
        
        assert portfolio.long_exposure == 50000
        assert portfolio.short_exposure == 30000
        assert portfolio.gross_exposure == 80000
        assert portfolio.net_exposure == 20000  # 50000 - 30000

    def test_rebalance_to_weights(self):
        """Test rebalancing to target weights."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        prices = {"BTC": 50000, "ETH": 3000}
        target_weights = {"BTC": 0.5, "ETH": 0.3}
        
        trades = portfolio.rebalance_to_weights(target_weights, prices, timestamp)
        
        # Should have opened positions
        weights = portfolio.get_weights()
        assert abs(weights.get("BTC", 0) - 0.5) < 0.05
        assert abs(weights.get("ETH", 0) - 0.3) < 0.05

    def test_close_all_positions(self):
        """Test closing all positions."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        portfolio.open_position("ETH", TradeDirection.LONG, 10.0, 3000, timestamp)
        
        prices = {"BTC": 52000, "ETH": 3100}
        total_pnl = portfolio.close_all_positions(prices, timestamp)
        
        # Should have no open positions
        positions = portfolio.get_positions()
        assert len(positions) == 0
        
        # PnL should be positive (prices went up)
        assert total_pnl > 0

    def test_take_snapshot(self):
        """Test taking portfolio snapshot."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        portfolio.update_prices({"BTC": 55000})
        
        snapshot = portfolio.take_snapshot(timestamp)
        
        assert snapshot.timestamp == timestamp
        assert snapshot.nav == 105000
        assert "BTC" in snapshot.positions

    def test_insufficient_cash_error(self):
        """Test error when insufficient cash."""
        portfolio = Portfolio(initial_capital=10_000)
        timestamp = datetime.now()
        
        with pytest.raises(ValueError, match="Insufficient cash"):
            portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)


# =============================================================================
# OrderExecutor Tests
# =============================================================================

class TestOrderExecutor:
    """Tests for OrderExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create executor with portfolio."""
        portfolio = Portfolio(initial_capital=100_000)
        cost_config = CostConfigModel(
            taker_rate=0.001,
            slippage_rate=0.0005,
        )
        return OrderExecutor(portfolio, cost_config)

    def test_create_order(self, executor: OrderExecutor):
        """Test creating an order."""
        order = executor.create_order(
            asset="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000,
        )
        
        assert order.order_id is not None
        assert order.asset == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING

    def test_execute_buy_order(self, executor: OrderExecutor):
        """Test executing a buy order."""
        order = executor.create_order(
            asset="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000,
        )
        
        success = executor.execute_order(order)
        
        assert success
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert order.filled_price > 50000  # Slippage applied

    def test_execute_sell_order(self, executor: OrderExecutor):
        """Test executing a sell order."""
        # First buy
        executor.execute_market_order(
            "BTCUSDT", OrderSide.BUY, 1.0, 50000, TradeDirection.LONG
        )
        
        # Then sell
        order = executor.create_order(
            asset="BTCUSDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=55000,
            direction=TradeDirection.LONG,
        )
        
        success = executor.execute_order(order)
        
        assert success
        assert order.status == OrderStatus.FILLED

    def test_rejected_order(self, executor: OrderExecutor):
        """Test order rejection for insufficient funds."""
        # Try to buy more than we can afford
        order = executor.create_order(
            asset="BTCUSDT",
            side=OrderSide.BUY,
            quantity=10.0,
            price=50000,  # 500k total, but only 100k available
        )
        
        success = executor.execute_order(order)
        
        assert not success
        assert order.status == OrderStatus.REJECTED

    def test_cancel_order(self, executor: OrderExecutor):
        """Test cancelling an order."""
        order = executor.create_order(
            asset="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000,
        )
        
        success = executor.cancel_order(order.order_id)
        
        assert success
        assert order.status == OrderStatus.CANCELLED

    def test_open_long_convenience(self, executor: OrderExecutor):
        """Test open_long convenience method."""
        order = executor.open_long("BTCUSDT", 0.5, 50000)
        
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert executor.portfolio.long_exposure > 0

    def test_open_short_convenience(self, executor: OrderExecutor):
        """Test open_short convenience method."""
        order = executor.open_short("BTCUSDT", 0.5, 50000)
        
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert executor.portfolio.short_exposure > 0

    def test_slippage_applied(self, executor: OrderExecutor):
        """Test that slippage is applied correctly."""
        order = executor.create_order(
            asset="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000,
        )
        
        executor.execute_order(order)
        
        # Buy order should have higher fill price (slippage = 0.0005)
        expected_fill = 50000 * (1 + 0.0005)
        assert abs(order.filled_price - expected_fill) < 0.01

    def test_execution_stats(self, executor: OrderExecutor):
        """Test execution statistics."""
        # Execute some orders
        executor.open_long("BTC", 0.1, 50000)
        executor.open_long("ETH", 1.0, 3000)
        
        # Try a rejected order
        order = executor.create_order("BTC", OrderSide.BUY, 100.0, 50000)
        executor.execute_order(order)
        
        stats = executor.get_stats()
        
        assert stats["total_orders"] == 3
        assert stats["filled_orders"] == 2
        assert stats["rejected_orders"] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestPortfolioIntegration:
    """Integration tests for portfolio workflow."""

    def test_complete_trade_cycle(self):
        """Test complete cycle: open -> update -> close."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        
        # Open position
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        initial_nav = portfolio.nav
        
        # Price goes up
        portfolio.update_prices({"BTC": 55000})
        assert portfolio.nav > initial_nav
        
        # Close with profit
        pnl = portfolio.close_position("BTC", TradeDirection.LONG, None, 55000, timestamp)
        
        assert pnl == 5000
        assert portfolio.total_realized_pnl == 5000
        assert portfolio.cash == 105000

    def test_long_short_portfolio(self):
        """Test portfolio with both long and short positions."""
        portfolio = Portfolio(initial_capital=200_000)
        timestamp = datetime.now()
        
        # Long BTC
        portfolio.open_position("BTC", TradeDirection.LONG, 1.0, 50000, timestamp)
        # Short ETH
        portfolio.open_position("ETH", TradeDirection.SHORT, 10.0, 3000, timestamp)
        
        # BTC goes up, ETH goes down (both profitable)
        portfolio.update_prices({"BTC": 55000, "ETH": 2800})
        
        # Long profit: 5000
        # Short profit: 10 * (3000 - 2800) = 2000
        assert portfolio.total_unrealized_pnl == 7000

    def test_rebalance_workflow(self):
        """Test rebalancing workflow."""
        portfolio = Portfolio(initial_capital=100_000)
        timestamp = datetime.now()
        prices = {"BTC": 50000, "ETH": 3000, "BNB": 300}
        
        # Initial allocation
        target1 = {"BTC": 0.4, "ETH": 0.3, "BNB": 0.2}
        portfolio.rebalance_to_weights(target1, prices, timestamp)
        
        # Verify allocation
        weights = portfolio.get_weights()
        assert abs(weights.get("BTC", 0) - 0.4) < 0.05
        
        # Rebalance to new target
        target2 = {"BTC": 0.5, "ETH": 0.5}  # No more BNB
        portfolio.rebalance_to_weights(target2, prices, timestamp)
        
        # BNB should be closed
        weights = portfolio.get_weights()
        assert "BNB" not in weights or abs(weights["BNB"]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
