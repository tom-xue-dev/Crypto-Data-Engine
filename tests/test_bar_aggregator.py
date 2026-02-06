"""
Unit tests for unified bar aggregator.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.services.bar_aggregator import (
    BarType,
    BarConfig,
    aggregate_bars,
    build_time_bars,
    build_tick_bars,
    build_volume_bars,
    build_dollar_bars,
    get_bar_builder,
    NUMBA_AVAILABLE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_tick_data():
    """Create sample tick data for testing."""
    np.random.seed(42)
    n_ticks = 10000
    
    # Generate timestamps (milliseconds)
    base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    timestamps = base_ts + np.cumsum(np.random.exponential(100, n_ticks)).astype(int)
    
    # Generate prices (random walk)
    price_changes = np.random.randn(n_ticks) * 0.1
    prices = 100 * np.exp(np.cumsum(price_changes / 100))
    
    # Generate quantities
    quantities = np.random.exponential(10, n_ticks)
    
    # Generate buyer maker flags
    is_buyer_maker = np.random.choice([True, False], n_ticks)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "quantity": quantities,
        "isBuyerMaker": is_buyer_maker,
    })


@pytest.fixture
def small_tick_data():
    """Create small tick data for quick tests."""
    np.random.seed(42)
    
    return pd.DataFrame({
        "timestamp": [1704067200000 + i * 1000 for i in range(100)],
        "price": 100 + np.random.randn(100) * 0.5,
        "quantity": np.random.exponential(10, 100),
        "isBuyerMaker": np.random.choice([True, False], 100),
    })


# =============================================================================
# BarType Tests
# =============================================================================

class TestBarType:
    """Tests for BarType enum."""

    def test_bar_type_values(self):
        """Test that all bar types are defined."""
        assert BarType.TIME_BAR.value == "time_bar"
        assert BarType.TICK_BAR.value == "tick_bar"
        assert BarType.VOLUME_BAR.value == "volume_bar"
        assert BarType.DOLLAR_BAR.value == "dollar_bar"

    def test_bar_type_from_string(self):
        """Test creating BarType from string."""
        assert BarType("time_bar") == BarType.TIME_BAR
        assert BarType("dollar_bar") == BarType.DOLLAR_BAR


# =============================================================================
# Unified Aggregation Tests
# =============================================================================

class TestAggregateBars:
    """Tests for aggregate_bars function."""

    def test_dollar_bars_basic(self, sample_tick_data):
        """Test basic dollar bar aggregation."""
        bars = aggregate_bars(
            sample_tick_data,
            "dollar_bar",
            threshold=10000,
            use_numba=False,
        )
        
        assert len(bars) > 0
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns

    def test_volume_bars_basic(self, sample_tick_data):
        """Test basic volume bar aggregation."""
        bars = aggregate_bars(
            sample_tick_data,
            "volume_bar",
            threshold=1000,
            use_numba=False,
        )
        
        assert len(bars) > 0
        assert all(bars["volume"] >= 0)

    def test_tick_bars_basic(self, small_tick_data):
        """Test basic tick bar aggregation."""
        bars = aggregate_bars(
            small_tick_data,
            "tick_bar",
            threshold=10,
        )
        
        assert len(bars) > 0
        # Each bar should have about 10 ticks (except possibly the last)
        assert bars["tick_count"].iloc[0] == 10

    def test_time_bars_basic(self, sample_tick_data):
        """Test basic time bar aggregation."""
        bars = aggregate_bars(
            sample_tick_data,
            "time_bar",
            threshold="1s",
        )
        
        assert len(bars) > 0
        assert "start_time" in bars.columns
        assert "end_time" in bars.columns

    def test_include_advanced_features(self, small_tick_data):
        """Test that advanced features are included."""
        bars = aggregate_bars(
            small_tick_data,
            "tick_bar",
            threshold=20,
            include_advanced=True,
        )
        
        # Check for advanced features
        assert "price_std" in bars.columns
        assert "up_move_ratio" in bars.columns
        assert "reversals" in bars.columns

    def test_exclude_advanced_features(self, small_tick_data):
        """Test that advanced features can be excluded."""
        bars = aggregate_bars(
            small_tick_data,
            "tick_bar",
            threshold=20,
            include_advanced=False,
        )
        
        # Should not have advanced features
        assert "skewness" not in bars.columns
        assert "kurtosis" not in bars.columns


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_build_time_bars(self, sample_tick_data):
        """Test build_time_bars function."""
        bars = build_time_bars(sample_tick_data, interval="5s")
        
        assert len(bars) > 0
        assert "open" in bars.columns

    def test_build_tick_bars(self, small_tick_data):
        """Test build_tick_bars function."""
        bars = build_tick_bars(small_tick_data, n_ticks=25)
        
        assert len(bars) > 0
        assert bars["tick_count"].iloc[0] == 25

    def test_build_volume_bars(self, sample_tick_data):
        """Test build_volume_bars function."""
        bars = build_volume_bars(sample_tick_data, volume_threshold=500)
        
        assert len(bars) > 0

    def test_build_dollar_bars(self, sample_tick_data):
        """Test build_dollar_bars function."""
        bars = build_dollar_bars(sample_tick_data, dollar_threshold=5000)
        
        assert len(bars) > 0


# =============================================================================
# Bar Builder Tests
# =============================================================================

class TestBarBuilders:
    """Tests for specific bar builders."""

    def test_time_bar_builder(self, sample_tick_data):
        """Test TimeBarBuilder directly."""
        config = BarConfig(BarType.TIME_BAR, "1s")
        builder = get_bar_builder(config)
        bars = builder.build_bars(sample_tick_data)
        
        assert len(bars) > 0

    def test_tick_bar_builder(self, small_tick_data):
        """Test TickBarBuilder directly."""
        config = BarConfig(BarType.TICK_BAR, 10)
        builder = get_bar_builder(config)
        bars = builder.build_bars(small_tick_data)
        
        assert len(bars) == 10  # 100 ticks / 10 per bar = 10 bars

    def test_volume_bar_builder(self, sample_tick_data):
        """Test VolumeBarBuilder directly."""
        config = BarConfig(BarType.VOLUME_BAR, 500)
        builder = get_bar_builder(config)
        bars = builder.build_bars(sample_tick_data)
        
        assert len(bars) > 0

    def test_dollar_bar_builder(self, sample_tick_data):
        """Test DollarBarBuilder directly."""
        config = BarConfig(BarType.DOLLAR_BAR, 10000)
        builder = get_bar_builder(config)
        bars = builder.build_bars(sample_tick_data)
        
        assert len(bars) > 0


# =============================================================================
# OHLCV Validation Tests
# =============================================================================

class TestOHLCVValidation:
    """Tests to validate OHLCV values are correct."""

    def test_high_low_relationship(self, sample_tick_data):
        """Test that high >= low for all bars."""
        bars = aggregate_bars(sample_tick_data, "tick_bar", 100)
        
        assert all(bars["high"] >= bars["low"])

    def test_open_close_in_range(self, sample_tick_data):
        """Test that open and close are within high-low range."""
        bars = aggregate_bars(sample_tick_data, "tick_bar", 100)
        
        assert all(bars["open"] >= bars["low"])
        assert all(bars["open"] <= bars["high"])
        assert all(bars["close"] >= bars["low"])
        assert all(bars["close"] <= bars["high"])

    def test_volume_positive(self, sample_tick_data):
        """Test that volume is always positive."""
        bars = aggregate_bars(sample_tick_data, "tick_bar", 100)
        
        assert all(bars["volume"] > 0)

    def test_buy_sell_volume_sum(self, sample_tick_data):
        """Test that buy + sell volume equals total volume."""
        bars = aggregate_bars(sample_tick_data, "tick_bar", 100, include_advanced=True)
        
        total = bars["buy_volume"] + bars["sell_volume"]
        assert np.allclose(total, bars["volume"], rtol=1e-10)


# =============================================================================
# Numba Acceleration Tests
# =============================================================================

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestNumbaAcceleration:
    """Tests for Numba-accelerated aggregation."""

    def test_dollar_bars_with_numba(self, sample_tick_data):
        """Test dollar bar aggregation with Numba."""
        bars = aggregate_bars(
            sample_tick_data,
            "dollar_bar",
            threshold=10000,
            use_numba=True,
        )
        
        assert len(bars) > 0
        assert "open" in bars.columns

    def test_volume_bars_with_numba(self, sample_tick_data):
        """Test volume bar aggregation with Numba."""
        bars = aggregate_bars(
            sample_tick_data,
            "volume_bar",
            threshold=1000,
            use_numba=True,
        )
        
        assert len(bars) > 0

    def test_numba_pandas_consistency(self, sample_tick_data):
        """Test that Numba and pandas produce similar results."""
        bars_numba = aggregate_bars(
            sample_tick_data,
            "dollar_bar",
            threshold=10000,
            use_numba=True,
        )
        
        bars_pandas = aggregate_bars(
            sample_tick_data,
            "dollar_bar",
            threshold=10000,
            use_numba=False,
        )
        
        # Should produce same number of bars (or very close)
        assert abs(len(bars_numba) - len(bars_pandas)) <= 1


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["timestamp", "price", "quantity", "isBuyerMaker"])
        
        bars = aggregate_bars(empty_df, "tick_bar", 10)
        
        assert len(bars) == 0

    def test_single_tick(self):
        """Test handling of single tick."""
        single_tick = pd.DataFrame({
            "timestamp": [1704067200000],
            "price": [100.0],
            "quantity": [10.0],
            "isBuyerMaker": [False],
        })
        
        bars = aggregate_bars(single_tick, "tick_bar", 10)
        
        assert len(bars) == 1
        assert bars["open"].iloc[0] == bars["close"].iloc[0]

    def test_missing_buyer_maker(self):
        """Test handling of missing isBuyerMaker column."""
        data = pd.DataFrame({
            "timestamp": [1704067200000 + i * 1000 for i in range(100)],
            "price": 100 + np.random.randn(100) * 0.5,
            "quantity": np.random.exponential(10, 100),
        })
        
        # Should still work
        bars = aggregate_bars(data, "tick_bar", 20)
        
        assert len(bars) > 0

    def test_invalid_bar_type(self):
        """Test handling of invalid bar type."""
        with pytest.raises(ValueError):
            aggregate_bars(pd.DataFrame(), "invalid_bar", 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
