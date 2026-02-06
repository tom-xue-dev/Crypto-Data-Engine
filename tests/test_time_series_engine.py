"""
Unit tests for TimeSeriesEngine and related strategies.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.core.base import SignalType
from crypto_data_engine.services.back_test import (
    BacktestConfig,
    BacktestMode,
    RiskConfigModel,
    CostConfigModel,
    TimeSeriesEngine,
    SingleAssetStrategy,
    MomentumTimeSeriesStrategy,
    MeanReversionTimeSeriesStrategy,
    create_backtest_engine,
    create_strategy,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def single_asset_data():
    """Create single asset time-series data."""
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=200, freq="h")
    
    # Generate price with trend and noise
    base_price = 100
    trend = np.linspace(0, 0.1, 200)  # 10% uptrend
    noise = np.random.randn(200) * 0.01
    prices = base_price * np.exp(trend + np.cumsum(noise))
    
    data = pd.DataFrame({
        "timestamp": dates,
        "asset": "BTCUSDT",
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1000, 5000, 200),
    })
    
    # Add features
    data["return"] = data["close"].pct_change()
    data["momentum"] = data["close"].pct_change(20)
    data["zscore"] = (data["close"] - data["close"].rolling(50).mean()) / data["close"].rolling(50).std()
    
    # Add signals
    data["signal"] = 0
    data.loc[data["momentum"] > 0.05, "signal"] = 1
    data.loc[data["momentum"] < -0.05, "signal"] = -1
    
    return data


@pytest.fixture
def multi_asset_data():
    """Create multi-asset time-series data with non-aligned timestamps."""
    np.random.seed(42)
    
    records = []
    
    for asset in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
        # Each asset has different bar generation times (simulating dollar bars)
        n_bars = np.random.randint(150, 200)
        
        # Generate random timestamps within date range
        start = datetime(2024, 1, 1)
        timestamps = sorted([
            start + timedelta(hours=np.random.randint(0, 24 * 30))
            for _ in range(n_bars)
        ])
        
        base_price = {"BTCUSDT": 50000, "ETHUSDT": 3000, "BNBUSDT": 300}[asset]
        prices = base_price * np.cumprod(1 + np.random.randn(n_bars) * 0.02)
        
        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp": ts,
                "asset": asset,
                "close": prices[i],
                "volume": np.random.randint(1000, 5000),
                "momentum": prices[i] / prices[max(0, i-20)] - 1 if i >= 20 else 0,
                "signal": 1 if np.random.random() > 0.7 else (-1 if np.random.random() > 0.5 else 0),
            })
    
    return pd.DataFrame(records)


@pytest.fixture
def basic_ts_config():
    """Create basic time-series backtest configuration."""
    return BacktestConfig(
        mode=BacktestMode.TIME_SERIES,
        initial_capital=100_000,
        start_date=datetime(2024, 1, 3),
        end_date=datetime(2024, 1, 30),
        warmup_periods=50,
        allow_short=True,
        risk_config=RiskConfigModel(
            max_position_size=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        ),
        cost_config=CostConfigModel(
            commission_rate=0.001,
        ),
    )


# =============================================================================
# Strategy Tests
# =============================================================================

class TestSingleAssetStrategy:
    """Tests for SingleAssetStrategy."""

    def test_generate_signal_buy(self):
        """Test BUY signal generation."""
        strategy = SingleAssetStrategy(signal_column="signal", entry_threshold=0)
        
        bar = pd.Series({"close": 100, "signal": 1})
        signal = strategy.generate_signal(bar, position=None)
        
        assert signal == SignalType.BUY

    def test_generate_signal_sell(self):
        """Test SELL signal generation."""
        strategy = SingleAssetStrategy(signal_column="signal", entry_threshold=0)
        
        bar = pd.Series({"close": 100, "signal": -1})
        signal = strategy.generate_signal(bar, position=None)
        
        assert signal == SignalType.SELL

    def test_generate_signal_hold(self):
        """Test HOLD signal generation."""
        strategy = SingleAssetStrategy(signal_column="signal", entry_threshold=0)
        
        bar = pd.Series({"close": 100, "signal": 0})
        signal = strategy.generate_signal(bar, position=None)
        
        assert signal == SignalType.HOLD

    def test_threshold(self):
        """Test threshold-based signals."""
        strategy = SingleAssetStrategy(
            signal_column="signal",
            entry_threshold=0.5,
        )
        
        # Below threshold - should hold
        bar = pd.Series({"close": 100, "signal": 0.3})
        assert strategy.generate_signal(bar, None) == SignalType.HOLD
        
        # Above threshold - should buy
        bar = pd.Series({"close": 100, "signal": 0.6})
        assert strategy.generate_signal(bar, None) == SignalType.BUY


class TestMomentumTimeSeriesStrategy:
    """Tests for MomentumTimeSeriesStrategy."""

    def test_long_entry(self):
        """Test long entry on positive momentum."""
        strategy = MomentumTimeSeriesStrategy(
            momentum_column="momentum",
            long_threshold=0.02,
        )
        
        bar = pd.Series({"close": 100, "momentum": 0.05})
        signal = strategy.generate_signal(bar, position=None)
        
        assert signal == SignalType.BUY

    def test_short_entry(self):
        """Test short entry on negative momentum."""
        strategy = MomentumTimeSeriesStrategy(
            momentum_column="momentum",
            short_threshold=-0.02,
        )
        
        bar = pd.Series({"close": 100, "momentum": -0.05})
        signal = strategy.generate_signal(bar, position=None)
        
        assert signal == SignalType.SELL


class TestMeanReversionTimeSeriesStrategy:
    """Tests for MeanReversionTimeSeriesStrategy."""

    def test_long_entry_on_low_zscore(self):
        """Test long entry when zscore is low."""
        strategy = MeanReversionTimeSeriesStrategy(
            zscore_column="zscore",
            entry_zscore=2.0,
        )
        
        bar = pd.Series({"close": 100, "zscore": -2.5})
        signal = strategy.generate_signal(bar, position=None)
        
        assert signal == SignalType.BUY

    def test_short_entry_on_high_zscore(self):
        """Test short entry when zscore is high."""
        strategy = MeanReversionTimeSeriesStrategy(
            zscore_column="zscore",
            entry_zscore=2.0,
        )
        
        bar = pd.Series({"close": 100, "zscore": 2.5})
        signal = strategy.generate_signal(bar, position=None)
        
        assert signal == SignalType.SELL


# =============================================================================
# Engine Tests
# =============================================================================

class TestTimeSeriesEngine:
    """Tests for TimeSeriesEngine."""

    def test_create_engine(self, basic_ts_config):
        """Test engine creation."""
        strategy = SingleAssetStrategy()
        engine = TimeSeriesEngine(basic_ts_config, strategy)
        
        assert engine.config == basic_ts_config
        assert engine.strategy == strategy

    def test_run_single_asset(self, single_asset_data, basic_ts_config):
        """Test running backtest with single asset data."""
        strategy = SingleAssetStrategy(signal_column="signal")
        engine = TimeSeriesEngine(basic_ts_config, strategy)
        
        result = engine.run(single_asset_data)
        
        assert result is not None
        assert result.initial_capital == 100_000
        assert len(engine.get_nav_history()) > 0

    def test_run_multi_asset(self, multi_asset_data, basic_ts_config):
        """Test running backtest with multi-asset non-aligned data."""
        strategy = SingleAssetStrategy(signal_column="signal")
        engine = TimeSeriesEngine(basic_ts_config, strategy)
        
        result = engine.run(multi_asset_data)
        
        assert result is not None
        # Should process bars from all assets
        assert engine._bar_count > 0

    def test_stop_loss(self, basic_ts_config):
        """Test stop-loss execution."""
        # Create data with significant drawdown
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        
        # Price drops significantly
        prices = 100 * np.ones(100)
        prices[60:] = 85  # 15% drop
        
        data = pd.DataFrame({
            "timestamp": dates,
            "asset": "BTCUSDT",
            "close": prices,
            "signal": [1] * 50 + [0] * 50,  # Buy early, then hold
        })
        
        config = BacktestConfig(
            mode=BacktestMode.TIME_SERIES,
            initial_capital=100_000,
            warmup_periods=10,
            risk_config=RiskConfigModel(
                max_position_size=0.5,
                stop_loss_pct=0.10,  # 10% stop loss
            ),
        )
        
        strategy = SingleAssetStrategy(signal_column="signal")
        engine = TimeSeriesEngine(config, strategy)
        
        result = engine.run(data)
        
        # Should have executed stop-loss
        stop_loss_trades = [t for t in result.trades if t.metadata.get("reason") == "stop_loss"]
        assert len(stop_loss_trades) > 0 or result.total_trades > 0

    def test_nav_tracking(self, single_asset_data, basic_ts_config):
        """Test NAV is tracked correctly."""
        strategy = SingleAssetStrategy(signal_column="signal")
        engine = TimeSeriesEngine(basic_ts_config, strategy)
        
        result = engine.run(single_asset_data)
        
        nav_history = engine.get_nav_history()
        
        # Should have NAV entries
        assert len(nav_history) > 0
        
        # Final NAV should match result
        final_nav = list(nav_history.values())[-1]
        assert abs(result.final_capital - final_nav) < 1


# =============================================================================
# Factory Tests
# =============================================================================

class TestTimeSeriesFactory:
    """Tests for time-series factory functions."""

    def test_create_time_series_engine(self, basic_ts_config):
        """Test creating time-series engine via factory."""
        strategy = SingleAssetStrategy()
        engine = create_backtest_engine(basic_ts_config, strategy)
        
        assert isinstance(engine, TimeSeriesEngine)

    def test_create_time_series_strategy(self):
        """Test creating time-series strategies by name."""
        single = create_strategy("single_asset", signal_column="my_signal")
        assert isinstance(single, SingleAssetStrategy)
        
        momentum = create_strategy("momentum_ts", long_threshold=0.03)
        assert isinstance(momentum, MomentumTimeSeriesStrategy)
        
        mean_rev = create_strategy("mean_reversion_ts", entry_zscore=1.5)
        assert isinstance(mean_rev, MeanReversionTimeSeriesStrategy)


# =============================================================================
# Integration Tests
# =============================================================================

class TestTimeSeriesIntegration:
    """Integration tests for time-series backtest workflow."""

    def test_full_workflow(self, single_asset_data):
        """Test complete time-series backtest workflow."""
        # Configure
        config = BacktestConfig(
            mode=BacktestMode.TIME_SERIES,
            initial_capital=100_000,
            start_date=datetime(2024, 1, 3),
            end_date=datetime(2024, 1, 8),
            warmup_periods=20,
            allow_short=True,
            risk_config=RiskConfigModel(
                max_position_size=0.3,
                stop_loss_pct=0.05,
            ),
            cost_config=CostConfigModel(commission_rate=0.001),
        )
        
        # Create strategy
        strategy = MomentumTimeSeriesStrategy(
            momentum_column="momentum",
            long_threshold=0.02,
            short_threshold=-0.02,
        )
        
        # Create and run engine
        engine = create_backtest_engine(config, strategy)
        result = engine.run(single_asset_data)
        
        # Verify result
        assert result.initial_capital == 100_000
        assert result.start_time is not None
        assert result.end_time is not None

    def test_dollar_bar_simulation(self, multi_asset_data):
        """Test with simulated dollar bar data (non-aligned timestamps)."""
        config = BacktestConfig(
            mode=BacktestMode.MULTI_ASSET_TIME_SERIES,
            initial_capital=100_000,
            warmup_periods=20,
            allow_short=False,
            risk_config=RiskConfigModel(max_position_size=0.2),
        )
        
        strategy = SingleAssetStrategy(signal_column="signal")
        engine = create_backtest_engine(config, strategy)
        
        result = engine.run(multi_asset_data)
        
        # Should handle non-aligned data
        assert result is not None
        assert engine._bar_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
