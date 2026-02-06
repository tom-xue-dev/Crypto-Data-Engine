"""
Unit tests for asset selector module.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.services.back_test import (
    AssetSelector,
    SelectionFilter,
    SelectionCriteria,
    VolumeBasedSelector,
    VolatilityBasedSelector,
    CompositeSelector,
    calculate_selection_metrics,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_asset_data():
    """Create sample asset data for testing."""
    np.random.seed(42)
    
    assets = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
        "XRPUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT",
    ]
    
    data = pd.DataFrame({
        "close": [50000, 3000, 300, 100, 0.5, 0.6, 5, 15, 0.8, 30],
        "volume": [1000, 5000, 3000, 8000, 10000, 9000, 4000, 2000, 7000, 6000],
        "dollar_volume": [50000000, 15000000, 900000, 800000, 5000, 5400, 20000, 30000, 5600, 180000],
        "volatility": [0.02, 0.03, 0.04, 0.06, 0.08, 0.07, 0.05, 0.04, 0.09, 0.05],
        "momentum": [0.10, -0.05, 0.03, -0.08, 0.02, 0.01, -0.02, 0.04, -0.03, 0.06],
    }, index=assets)
    
    return data


@pytest.fixture
def sample_timeseries_data():
    """Create sample time-series data."""
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
    
    records = []
    for asset in assets:
        base_price = {"BTCUSDT": 50000, "ETHUSDT": 3000, "BNBUSDT": 300, 
                      "SOLUSDT": 100, "ADAUSDT": 0.5}[asset]
        base_vol = {"BTCUSDT": 1000, "ETHUSDT": 5000, "BNBUSDT": 3000,
                    "SOLUSDT": 8000, "ADAUSDT": 10000}[asset]
        
        for date in dates:
            records.append({
                "timestamp": date,
                "asset": asset,
                "close": base_price * (1 + np.random.randn() * 0.02),
                "volume": base_vol * (1 + np.random.randn() * 0.1),
            })
    
    df = pd.DataFrame(records)
    df["dollar_volume"] = df["close"] * df["volume"]
    
    return df.set_index(["timestamp", "asset"])


# =============================================================================
# SelectionFilter Tests
# =============================================================================

class TestSelectionFilter:
    """Tests for SelectionFilter."""

    def test_get_column_name_default(self):
        """Test default column name from criteria."""
        filter_cfg = SelectionFilter(criteria=SelectionCriteria.DOLLAR_VOLUME)
        assert filter_cfg.get_column_name() == "dollar_volume"

    def test_get_column_name_custom(self):
        """Test custom column name."""
        filter_cfg = SelectionFilter(
            criteria=SelectionCriteria.PRICE_RANGE,
            column="close",
        )
        assert filter_cfg.get_column_name() == "close"


# =============================================================================
# AssetSelector Tests
# =============================================================================

class TestAssetSelector:
    """Tests for AssetSelector."""

    def test_basic_selection(self, sample_asset_data):
        """Test basic asset selection."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    top_n=5,
                ),
            ],
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        assert len(selected) == 5
        # Top 5 by dollar volume should include BTC and ETH
        assert "BTCUSDT" in selected
        assert "ETHUSDT" in selected

    def test_multiple_filters(self, sample_asset_data):
        """Test multiple filters with AND logic."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    top_n=8,
                ),
                SelectionFilter(
                    criteria=SelectionCriteria.VOLATILITY,
                    max_value=0.05,
                ),
            ],
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # All selected should have volatility <= 0.05
        for asset in selected:
            if asset in sample_asset_data.index:
                assert sample_asset_data.loc[asset, "volatility"] <= 0.05

    def test_include_assets(self, sample_asset_data):
        """Test that include_assets are always selected."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    top_n=3,
                ),
            ],
            include_assets=["ADAUSDT"],  # Low volume but required
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        assert "ADAUSDT" in selected

    def test_exclude_assets(self, sample_asset_data):
        """Test that exclude_assets are never selected."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    top_n=5,
                ),
            ],
            exclude_assets=["BTCUSDT"],  # Highest volume but excluded
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        assert "BTCUSDT" not in selected

    def test_min_value_filter(self, sample_asset_data):
        """Test minimum value filter."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.PRICE_RANGE,
                    column="close",
                    min_value=10.0,
                ),
            ],
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # All selected should have close >= 10
        for asset in selected:
            assert sample_asset_data.loc[asset, "close"] >= 10.0

    def test_max_value_filter(self, sample_asset_data):
        """Test maximum value filter."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.VOLATILITY,
                    max_value=0.05,
                ),
            ],
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # All selected should have volatility <= 0.05
        for asset in selected:
            assert sample_asset_data.loc[asset, "volatility"] <= 0.05

    def test_percentile_filter(self, sample_asset_data):
        """Test percentile-based filter."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    percentile_min=50,  # Top 50%
                ),
            ],
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # Should have about half the assets
        assert len(selected) >= 4

    def test_reselect_frequency_daily(self, sample_asset_data):
        """Test daily reselection frequency."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    top_n=5,
                ),
            ],
            reselect_frequency="D",
        )
        
        # First selection
        selected1 = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # Same day - should use cache
        selected2 = selector.select(sample_asset_data, datetime(2024, 1, 1, 12, 0))
        
        # Next day - should reselect
        selected3 = selector.select(sample_asset_data, datetime(2024, 1, 2))
        
        assert selected1 == selected2  # Same day, cached
        assert len(selected3) > 0  # Reselected

    def test_max_assets_limit(self, sample_asset_data):
        """Test maximum assets limit."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.PRICE_RANGE,
                    column="close",
                    min_value=0.1,  # Most assets pass
                ),
            ],
            max_assets=3,
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        assert len(selected) <= 3

    def test_selection_history(self, sample_asset_data):
        """Test selection history tracking."""
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    top_n=5,
                ),
            ],
            reselect_frequency="D",
        )
        
        selector.select(sample_asset_data, datetime(2024, 1, 1))
        selector.select(sample_asset_data, datetime(2024, 1, 2), force_reselect=True)
        
        history = selector.get_selection_history()
        
        assert len(history) == 2


# =============================================================================
# Pre-configured Selector Tests
# =============================================================================

class TestVolumeBasedSelector:
    """Tests for VolumeBasedSelector."""

    def test_basic_selection(self, sample_asset_data):
        """Test volume-based selection."""
        selector = VolumeBasedSelector(top_n=3)
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        assert len(selected) == 3
        # BTC has highest dollar volume
        assert "BTCUSDT" in selected


class TestVolatilityBasedSelector:
    """Tests for VolatilityBasedSelector."""

    def test_volatility_range(self, sample_asset_data):
        """Test volatility range selection."""
        selector = VolatilityBasedSelector(
            min_volatility=0.03,
            max_volatility=0.06,
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # All selected should have volatility in range
        for asset in selected:
            vol = sample_asset_data.loc[asset, "volatility"]
            assert 0.03 <= vol <= 0.06


class TestCompositeSelector:
    """Tests for CompositeSelector."""

    def test_default_selector(self, sample_asset_data):
        """Test default composite selector."""
        selector = CompositeSelector.create_default(
            top_n_volume=5,
            min_price=1.0,
            max_volatility=0.06,
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # Should apply all filters
        for asset in selected:
            # Check price
            assert sample_asset_data.loc[asset, "close"] >= 1.0
            # Check volatility
            assert sample_asset_data.loc[asset, "volatility"] <= 0.06


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestCalculateSelectionMetrics:
    """Tests for calculate_selection_metrics function."""

    def test_adds_metrics(self, sample_timeseries_data):
        """Test that metrics are calculated."""
        result = calculate_selection_metrics(sample_timeseries_data, lookback=10)
        
        # Should have volatility
        assert "volatility" in result.columns
        
        # Should have momentum
        assert "momentum" in result.columns

    def test_handles_multiindex(self, sample_timeseries_data):
        """Test handling of multi-indexed data."""
        result = calculate_selection_metrics(sample_timeseries_data, lookback=5)
        
        # Should preserve structure
        assert not result.empty


# =============================================================================
# Integration Tests
# =============================================================================

class TestAssetSelectorIntegration:
    """Integration tests for asset selection workflow."""

    def test_full_workflow(self, sample_timeseries_data):
        """Test complete asset selection workflow."""
        # 1. Calculate metrics
        data_with_metrics = calculate_selection_metrics(
            sample_timeseries_data, lookback=10
        )
        
        # 2. Create selector
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    top_n=3,
                ),
            ],
            include_assets=["BTCUSDT"],
            reselect_frequency="W",
        )
        
        # 3. Get latest cross-section
        latest_time = data_with_metrics.index.get_level_values(0).max()
        latest_data = data_with_metrics.loc[latest_time]
        
        # 4. Select assets
        selected = selector.select(latest_data, latest_time)
        
        # Verify
        assert len(selected) >= 1
        assert "BTCUSDT" in selected  # Required include

    def test_with_backtest_config(self, sample_asset_data):
        """Test integration with backtest configuration."""
        # Create selector matching backtest config requirements
        selector = CompositeSelector.create_default(
            top_n_volume=50,
            min_price=0.01,
            max_price=100000,
            max_volatility=0.1,
        )
        
        selected = selector.select(sample_asset_data, datetime(2024, 1, 1))
        
        # Should return valid asset list
        assert isinstance(selected, list)
        assert all(isinstance(a, str) for a in selected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
