"""
Unit tests for unified feature calculator.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.services.feature import (
    UnifiedFeatureConfig,
    UnifiedFeatureCalculator,
    calculate_features,
    calculate_features_multi_asset,
    select_features_by_correlation,
    get_feature_importance,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bar_data():
    """Create sample bar data for testing."""
    np.random.seed(42)
    n = 500
    
    # Generate price series
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        "start_time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": prices * (1 - np.abs(np.random.randn(n) * 0.001)),
        "high": prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        "low": prices * (1 - np.abs(np.random.randn(n) * 0.005)),
        "close": prices,
        "volume": np.random.exponential(1000, n),
        "buy_volume": np.random.exponential(500, n),
        "sell_volume": np.random.exponential(500, n),
        "vwap": prices * (1 + np.random.randn(n) * 0.001),
        "tick_count": np.random.randint(100, 1000, n),
        "dollar_volume": prices * np.random.exponential(1000, n),
        "up_move_ratio": np.random.uniform(0.3, 0.7, n),
        "reversals": np.random.randint(5, 50, n),
    })
    
    # Fix volume sum
    data["volume"] = data["buy_volume"] + data["sell_volume"]
    
    return data


@pytest.fixture
def multi_asset_data(sample_bar_data):
    """Create multi-asset bar data."""
    assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    dfs = []
    for asset in assets:
        df = sample_bar_data.copy()
        # Add some asset-specific variation
        df["close"] = df["close"] * (1 + np.random.randn(len(df)) * 0.01)
        df["asset"] = asset
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def small_bar_data():
    """Create small bar data for quick tests."""
    np.random.seed(42)
    n = 50
    
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    return pd.DataFrame({
        "close": prices,
        "volume": np.random.exponential(1000, n),
    })


# =============================================================================
# Configuration Tests
# =============================================================================

class TestUnifiedFeatureConfig:
    """Tests for UnifiedFeatureConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedFeatureConfig()
        
        assert config.windows == [5, 10, 20, 60, 120]
        assert config.include_returns is True
        assert config.include_volatility is True
        assert config.normalize is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = UnifiedFeatureConfig(
            windows=[5, 10],
            include_alphas=False,
            normalize=True,
        )
        
        assert config.windows == [5, 10]
        assert config.include_alphas is False
        assert config.normalize is True


# =============================================================================
# Feature Calculation Tests
# =============================================================================

class TestUnifiedFeatureCalculator:
    """Tests for UnifiedFeatureCalculator."""

    def test_basic_calculation(self, sample_bar_data):
        """Test basic feature calculation."""
        calculator = UnifiedFeatureCalculator()
        result = calculator.calculate(sample_bar_data)
        
        # Should have more columns than input
        assert len(result.columns) > len(sample_bar_data.columns)
        
        # Should have same number of rows
        assert len(result) == len(sample_bar_data)

    def test_returns_calculated(self, sample_bar_data):
        """Test that return features are calculated."""
        config = UnifiedFeatureConfig(
            windows=[5, 10],
            include_returns=True,
            include_volatility=False,
            include_momentum=False,
            include_volume=False,
            include_microstructure=False,
            include_alphas=False,
            include_technical=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate(sample_bar_data)
        
        assert "return_1" in result.columns
        assert "return_5" in result.columns
        assert "return_10" in result.columns

    def test_volatility_calculated(self, sample_bar_data):
        """Test that volatility features are calculated."""
        config = UnifiedFeatureConfig(
            windows=[10, 20],
            include_returns=True,
            include_volatility=True,
            include_momentum=False,
            include_volume=False,
            include_microstructure=False,
            include_alphas=False,
            include_technical=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate(sample_bar_data)
        
        assert "volatility_10" in result.columns
        assert "volatility_20" in result.columns

    def test_momentum_calculated(self, sample_bar_data):
        """Test that momentum features are calculated."""
        config = UnifiedFeatureConfig(
            windows=[10, 20],
            include_returns=False,
            include_volatility=False,
            include_momentum=True,
            include_volume=False,
            include_microstructure=False,
            include_alphas=False,
            include_technical=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate(sample_bar_data)
        
        assert "momentum_10" in result.columns
        assert "sma_10" in result.columns
        assert "rsi_14" in result.columns

    def test_volume_features(self, sample_bar_data):
        """Test volume feature calculation."""
        config = UnifiedFeatureConfig(
            windows=[10],
            include_returns=False,
            include_volatility=False,
            include_momentum=False,
            include_volume=True,
            include_microstructure=False,
            include_alphas=False,
            include_technical=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate(sample_bar_data)
        
        assert "volume_sma_10" in result.columns
        assert "buy_ratio" in result.columns
        assert "imbalance" in result.columns

    def test_alpha_factors(self, sample_bar_data):
        """Test alpha factor calculation."""
        config = UnifiedFeatureConfig(
            windows=[60, 120],
            include_returns=True,
            include_volatility=False,
            include_momentum=False,
            include_volume=False,
            include_microstructure=False,
            include_alphas=True,
            include_technical=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate(sample_bar_data)
        
        # Should have at least some alpha features
        alpha_cols = [c for c in result.columns if "alpha" in c.lower()]
        assert len(alpha_cols) > 0

    def test_normalization(self, sample_bar_data):
        """Test feature normalization."""
        config = UnifiedFeatureConfig(
            windows=[10],
            include_returns=True,
            include_volatility=False,
            include_momentum=False,
            include_volume=False,
            include_microstructure=False,
            include_alphas=False,
            include_technical=False,
            normalize=True,
            winsorize_std=3.0,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate(sample_bar_data)
        
        # Normalized features should be within winsorize bounds
        return_col = result["return_10"].dropna()
        assert return_col.max() <= 3.0
        assert return_col.min() >= -3.0


# =============================================================================
# Multi-Asset Tests
# =============================================================================

class TestMultiAssetFeatures:
    """Tests for multi-asset feature calculation."""

    def test_multi_asset_basic(self, multi_asset_data):
        """Test basic multi-asset calculation."""
        config = UnifiedFeatureConfig(
            windows=[10, 20],
            include_cross_sectional=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate_multi_asset(multi_asset_data)
        
        # Should have all assets
        assert set(result["asset"].unique()) == {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

    def test_cross_sectional_features(self, multi_asset_data):
        """Test cross-sectional feature calculation."""
        config = UnifiedFeatureConfig(
            windows=[20],
            include_returns=True,
            include_volatility=True,
            include_cross_sectional=True,
            rank_columns=["return_20", "volatility_20"],
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate_multi_asset(multi_asset_data)
        
        # Should have rank and zscore columns
        assert "return_20_rank" in result.columns
        assert "return_20_zscore" in result.columns


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_features(self, sample_bar_data):
        """Test calculate_features function."""
        result = calculate_features(
            sample_bar_data,
            windows=[10, 20],
            include_alphas=False,
        )
        
        assert len(result.columns) > len(sample_bar_data.columns)

    def test_calculate_features_multi_asset(self, multi_asset_data):
        """Test calculate_features_multi_asset function."""
        result = calculate_features_multi_asset(
            multi_asset_data,
            windows=[10],
            include_cross_sectional=True,
        )
        
        assert "asset" in result.columns


# =============================================================================
# Feature Selection Tests
# =============================================================================

class TestFeatureSelection:
    """Tests for feature selection utilities."""

    def test_select_by_correlation(self, sample_bar_data):
        """Test feature selection by correlation."""
        features = calculate_features(
            sample_bar_data,
            windows=[10, 20],
            include_alphas=False,
            drop_na=True,
        )
        
        # Create target (next return)
        target = features["close"].pct_change().shift(-1).dropna()
        features = features.iloc[:-1]  # Align with target
        
        if len(features) > 0 and len(target) > 0:
            selected = select_features_by_correlation(
                features,
                target,
                top_n=10,
                min_correlation=0.01,
            )
            
            assert isinstance(selected, list)
            assert len(selected) <= 10

    def test_feature_importance(self, sample_bar_data):
        """Test feature importance calculation."""
        features = calculate_features(
            sample_bar_data,
            windows=[10, 20],
            include_alphas=False,
            drop_na=True,
        )
        
        target = features["close"].pct_change().shift(-1).dropna()
        features = features.iloc[:-1]
        
        if len(features) > 10:
            importance = get_feature_importance(
                features,
                target,
                method="correlation",
            )
            
            assert isinstance(importance, pd.Series)
            assert len(importance) > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimal_data(self, small_bar_data):
        """Test with minimal data."""
        result = calculate_features(
            small_bar_data,
            windows=[5, 10],
            include_alphas=False,
            include_microstructure=False,
            include_technical=False,
        )
        
        assert len(result) == len(small_bar_data)

    def test_missing_columns(self):
        """Test handling of missing columns."""
        data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
        })
        
        # Should not crash - use config directly for more control
        config = UnifiedFeatureConfig(
            windows=[2, 3],
            include_volume=False,
            include_microstructure=False,
            include_alphas=False,
            include_technical=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        result = calculator.calculate(data)
        
        assert "return_1" in result.columns

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        data = pd.DataFrame(columns=["close", "volume"])
        
        result = calculate_features(
            data,
            windows=[5],
            include_alphas=False,
            include_technical=False,
        )
        
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
