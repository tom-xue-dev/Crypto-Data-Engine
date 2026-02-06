"""
Unit tests for backtest configuration module.
"""
from datetime import datetime

import pytest

from crypto_data_engine.services.back_test.config import (
    BacktestConfig,
    BacktestMode,
    RiskConfigModel,
    CostConfigModel,
    AssetPoolConfig,
    StopLossStrategy,
    create_momentum_config,
    create_mean_reversion_config,
    create_dollar_bar_config,
)


# =============================================================================
# BacktestMode Tests
# =============================================================================

class TestBacktestMode:
    """Tests for BacktestMode enum."""

    def test_mode_values(self):
        """Test enum values."""
        assert BacktestMode.CROSS_SECTIONAL.value == "cross_sectional"
        assert BacktestMode.TIME_SERIES.value == "time_series"
        assert BacktestMode.MULTI_ASSET_TIME_SERIES.value == "multi_asset_time_series"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert BacktestMode("cross_sectional") == BacktestMode.CROSS_SECTIONAL
        assert BacktestMode("time_series") == BacktestMode.TIME_SERIES


# =============================================================================
# RiskConfigModel Tests
# =============================================================================

class TestRiskConfigModel:
    """Tests for RiskConfigModel."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RiskConfigModel()
        
        assert config.max_position_size == 0.1
        assert config.max_total_exposure == 1.0
        assert config.max_leverage == 1.0
        assert config.max_drawdown == 0.2
        assert config.stop_loss_enabled is True
        assert StopLossStrategy.FIXED in config.stop_loss_strategies

    def test_custom_values(self):
        """Test custom configuration."""
        config = RiskConfigModel(
            max_position_size=0.2,
            max_leverage=2.0,
            stop_loss_strategies=[StopLossStrategy.TRAILING, StopLossStrategy.ATR_BASED],
        )
        
        assert config.max_position_size == 0.2
        assert config.max_leverage == 2.0
        assert len(config.stop_loss_strategies) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RiskConfigModel(
            stop_loss_strategies=[StopLossStrategy.FIXED],
        )
        data = config.to_dict()
        
        assert "max_position_size" in data
        assert data["stop_loss_strategies"] == ["fixed"]

    def test_validation_bounds(self):
        """Test validation of bounded fields."""
        # max_position_size should be between 0 and 1
        with pytest.raises(ValueError):
            RiskConfigModel(max_position_size=1.5)
        
        with pytest.raises(ValueError):
            RiskConfigModel(max_position_size=-0.1)


# =============================================================================
# CostConfigModel Tests
# =============================================================================

class TestCostConfigModel:
    """Tests for CostConfigModel."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CostConfigModel()
        
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.0005
        assert config.funding_enabled is False

    def test_calculate_total_cost(self):
        """Test cost calculation."""
        config = CostConfigModel(
            taker_rate=0.001,
            slippage_rate=0.0005,
        )
        
        trade_value = 10000
        slippage, commission = config.calculate_total_cost(trade_value)
        
        assert commission == 10.0  # 10000 * 0.001
        assert slippage == 5.0  # 10000 * 0.0005

    def test_calculate_cost_maker(self):
        """Test cost calculation for maker order."""
        config = CostConfigModel(
            maker_rate=0.0002,
            taker_rate=0.0005,
        )
        
        trade_value = 10000
        _, commission_taker = config.calculate_total_cost(trade_value, is_maker=False)
        _, commission_maker = config.calculate_total_cost(trade_value, is_maker=True)
        
        assert commission_maker < commission_taker
        assert commission_maker == 2.0  # 10000 * 0.0002
        assert commission_taker == 5.0  # 10000 * 0.0005

    def test_min_commission(self):
        """Test minimum commission floor."""
        config = CostConfigModel(
            commission_rate=0.001,
            min_commission=1.0,
        )
        
        # Small trade should hit minimum
        _, commission = config.calculate_total_cost(100)
        assert commission == 1.0


# =============================================================================
# AssetPoolConfig Tests
# =============================================================================

class TestAssetPoolConfig:
    """Tests for AssetPoolConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = AssetPoolConfig()
        
        assert config.enabled is False
        assert config.top_n == 100
        assert "dollar_volume" in config.selection_criteria

    def test_custom_config(self):
        """Test custom configuration."""
        config = AssetPoolConfig(
            enabled=True,
            top_n=50,
            exclude_assets=["BTCUSDT"],
            include_assets=["ETHUSDT"],
        )
        
        assert config.enabled is True
        assert config.top_n == 50
        assert "BTCUSDT" in config.exclude_assets
        assert "ETHUSDT" in config.include_assets


# =============================================================================
# BacktestConfig Tests
# =============================================================================

class TestBacktestConfig:
    """Tests for main BacktestConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = BacktestConfig()
        
        assert config.mode == BacktestMode.CROSS_SECTIONAL
        assert config.initial_capital == 1_000_000
        assert config.risk_config is not None
        assert config.cost_config is not None

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            mode=BacktestMode.TIME_SERIES,
            initial_capital=500_000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            bar_type="dollar",
        )
        
        assert config.mode == BacktestMode.TIME_SERIES
        assert config.initial_capital == 500_000
        assert config.bar_type == "dollar"

    def test_cross_sectional_config(self):
        """Test cross-sectional specific configuration."""
        config = BacktestConfig(
            mode=BacktestMode.CROSS_SECTIONAL,
            rebalance_frequency="W-MON",
            top_n=20,
            top_n_long=10,
            top_n_short=10,
            ranking_factor="return_1w",
        )
        
        assert config.rebalance_frequency == "W-MON"
        assert config.top_n_long + config.top_n_short == config.top_n

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BacktestConfig(
            mode=BacktestMode.CROSS_SECTIONAL,
            start_date=datetime(2024, 1, 1),
        )
        data = config.to_dict()
        
        assert data["mode"] == "cross_sectional"
        assert data["start_date"] == "2024-01-01T00:00:00"
        assert "risk_config" in data

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "mode": "time_series",
            "initial_capital": 500000,
            "start_date": "2024-01-01T00:00:00",
            "risk_config": {"max_position_size": 0.15},
        }
        
        config = BacktestConfig.from_dict(data)
        
        assert config.mode == BacktestMode.TIME_SERIES
        assert config.initial_capital == 500000
        assert config.risk_config.max_position_size == 0.15

    def test_validation_valid_config(self):
        """Test validation with valid config."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            top_n=20,
            top_n_long=10,
            top_n_short=10,
        )
        
        messages = config.validate()
        assert len(messages) == 0

    def test_validation_invalid_dates(self):
        """Test validation catches invalid date range."""
        config = BacktestConfig(
            start_date=datetime(2024, 12, 31),
            end_date=datetime(2024, 1, 1),  # Before start!
        )
        
        messages = config.validate()
        assert any("start_date" in msg for msg in messages)

    def test_validation_invalid_top_n(self):
        """Test validation catches top_n overflow."""
        config = BacktestConfig(
            top_n=10,
            top_n_long=10,
            top_n_short=10,  # Sum > top_n
        )
        
        messages = config.validate()
        assert any("top_n" in msg for msg in messages)

    def test_validation_short_disabled(self):
        """Test validation catches short positions when disabled."""
        config = BacktestConfig(
            allow_short=False,
            top_n_short=5,
        )
        
        messages = config.validate()
        assert any("allow_short" in msg for msg in messages)

    def test_validation_negative_capital(self):
        """Test validation catches negative capital."""
        config = BacktestConfig(initial_capital=-1000)
        
        messages = config.validate()
        assert any("initial_capital" in msg for msg in messages)


# =============================================================================
# Preset Configuration Tests
# =============================================================================

class TestPresetConfigs:
    """Tests for preset configuration factories."""

    def test_momentum_config(self):
        """Test momentum strategy preset."""
        config = create_momentum_config(
            initial_capital=500_000,
            lookback_days=10,
            top_n=30,
        )
        
        assert config.mode == BacktestMode.CROSS_SECTIONAL
        assert config.initial_capital == 500_000
        assert config.top_n == 30
        assert config.top_n_long == 15
        assert config.top_n_short == 15
        assert "return_10d" in config.ranking_factor
        assert "Momentum" in config.name

    def test_mean_reversion_config(self):
        """Test mean reversion strategy preset."""
        config = create_mean_reversion_config(
            lookback_days=3,
            rebalance_freq="D",
        )
        
        assert config.mode == BacktestMode.CROSS_SECTIONAL
        assert config.rebalance_frequency == "D"
        assert "return_3d" in config.ranking_factor
        assert "Mean Reversion" in config.name

    def test_dollar_bar_config(self):
        """Test dollar bar strategy preset."""
        config = create_dollar_bar_config(
            initial_capital=100_000,
            dollar_threshold=500_000,
        )
        
        assert config.mode == BacktestMode.MULTI_ASSET_TIME_SERIES
        assert config.bar_type == "dollar"
        assert config.aligned is False
        assert "Dollar Bar" in config.name


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfigIntegration:
    """Integration tests for configuration workflows."""

    def test_roundtrip_serialization(self):
        """Test config survives round-trip serialization."""
        original = BacktestConfig(
            mode=BacktestMode.CROSS_SECTIONAL,
            initial_capital=1_000_000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            risk_config=RiskConfigModel(max_position_size=0.15),
            cost_config=CostConfigModel(commission_rate=0.0005),
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = BacktestConfig.from_dict(data)
        
        assert restored.mode == original.mode
        assert restored.initial_capital == original.initial_capital
        assert restored.risk_config.max_position_size == original.risk_config.max_position_size
        assert restored.cost_config.commission_rate == original.cost_config.commission_rate

    def test_config_with_all_options(self):
        """Test config with all options set."""
        config = BacktestConfig(
            mode=BacktestMode.CROSS_SECTIONAL,
            initial_capital=2_000_000,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
            bar_type="time",
            rebalance_frequency="W-FRI",
            top_n=50,
            top_n_long=25,
            top_n_short=25,
            risk_config=RiskConfigModel(
                max_position_size=0.05,
                max_drawdown=0.15,
                stop_loss_strategies=[StopLossStrategy.TRAILING],
            ),
            cost_config=CostConfigModel(
                taker_rate=0.0003,
                slippage_rate=0.0002,
            ),
            asset_pool_config=AssetPoolConfig(
                enabled=True,
                top_n=200,
            ),
            allow_short=True,
            log_trades=True,
            name="Full Featured Backtest",
        )
        
        # Should validate without errors
        messages = config.validate()
        assert len(messages) == 0
        
        # Should serialize correctly
        data = config.to_dict()
        assert data["name"] == "Full Featured Backtest"
        assert data["risk_config"]["max_position_size"] == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
