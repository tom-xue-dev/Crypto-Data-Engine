"""
End-to-end tests for the complete backtesting workflow.

Tests the full pipeline from data -> bars -> features -> signals -> backtest.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# Bar aggregation
from crypto_data_engine.services.bar_aggregator import (
    aggregate_bars,
    build_dollar_bars,
    BarType,
)

# Feature calculation
from crypto_data_engine.services.feature import (
    calculate_features,
    UnifiedFeatureConfig,
    UnifiedFeatureCalculator,
)

# Signal generation
from crypto_data_engine.services.signal_generation import (
    SignalOutput,
    FactorSignalGenerator,
    RankSignalGenerator,
    EnsembleSignalGenerator,
    EnsembleMethod,
    GeneratorConfig,
    FactorConfig,
)

# Asset selection
from crypto_data_engine.services.back_test.asset_selector import (
    AssetSelector,
    SelectionFilter,
    SelectionCriteria,
)

# Backtest engine
from crypto_data_engine.services.back_test import (
    BacktestConfig,
    BacktestMode,
    RiskConfigModel,
    CostConfigModel,
    CrossSectionalEngine,
    TimeSeriesEngine,
    MomentumStrategy,
    MeanReversionStrategy,
    create_backtest_engine,
    create_strategy,
)


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_tick_data(
    n_ticks: int = 50000,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic tick data."""
    np.random.seed(seed)
    
    # Timestamps (milliseconds)
    base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    timestamps = base_ts + np.cumsum(np.random.exponential(200, n_ticks)).astype(int)
    
    # Price process (random walk with drift)
    returns = np.random.randn(n_ticks) * volatility / np.sqrt(n_ticks)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some momentum
    prices = prices * (1 + 0.001 * np.sin(np.linspace(0, 10 * np.pi, n_ticks)))
    
    # Volume
    quantities = np.random.exponential(10, n_ticks)
    
    # Trade direction
    is_buyer_maker = np.random.choice([True, False], n_ticks)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "quantity": quantities,
        "isBuyerMaker": is_buyer_maker,
    })


def generate_multi_asset_bars(
    assets: List[str],
    n_days: int = 60,
    bars_per_day: int = 24,
) -> pd.DataFrame:
    """Generate multi-asset bar data."""
    all_data = []
    
    for i, asset in enumerate(assets):
        np.random.seed(42 + i)
        n_bars = n_days * bars_per_day
        
        # Generate timestamps
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=j) for j in range(n_bars)]
        
        # Generate prices with asset-specific drift
        drift = 0.0001 * (i - len(assets) / 2)  # Some assets up, some down
        volatility = 0.02 * (1 + i * 0.1)  # Different volatilities
        returns = drift + np.random.randn(n_bars) * volatility
        prices = 100 * (1 + i * 0.5) * np.exp(np.cumsum(returns))
        
        # OHLCV data
        highs = prices * (1 + np.abs(np.random.randn(n_bars) * 0.005))
        lows = prices * (1 - np.abs(np.random.randn(n_bars) * 0.005))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        volumes = np.random.exponential(1000, n_bars) * (1 + i * 0.2)
        dollar_volumes = prices * volumes
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "asset": asset,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "dollar_volume": dollar_volumes,
            "buy_volume": volumes * np.random.uniform(0.4, 0.6, n_bars),
            "sell_volume": volumes * np.random.uniform(0.4, 0.6, n_bars),
            "vwap": prices * (1 + np.random.randn(n_bars) * 0.001),
        })
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tick_data():
    """Generate tick data for tests."""
    return generate_tick_data(n_ticks=30000, volatility=0.015)


@pytest.fixture
def multi_asset_bars():
    """Generate multi-asset bar data."""
    return generate_multi_asset_bars(
        assets=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
        n_days=90,
        bars_per_day=24,
    )


# =============================================================================
# E2E Test: Tick -> Dollar Bars -> Features
# =============================================================================

class TestTickToFeatures:
    """Test pipeline from tick data to features."""

    def test_tick_to_bars(self, tick_data):
        """Test converting tick data to bars."""
        bars = aggregate_bars(
            tick_data,
            BarType.DOLLAR_BAR,
            threshold=50000,
            use_numba=True,
        )
        
        assert len(bars) > 0
        assert "open" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns

    def test_bars_to_features(self, tick_data):
        """Test converting bars to features."""
        # Step 1: Tick -> Bars
        bars = build_dollar_bars(tick_data, dollar_threshold=50000)
        
        # Step 2: Bars -> Features
        features = calculate_features(
            bars,
            windows=[5, 10, 20],
            include_alphas=False,
            include_technical=False,
        )
        
        assert len(features) == len(bars)
        assert "return_1" in features.columns
        assert "volatility_10" in features.columns
        assert "momentum_20" in features.columns

    def test_full_pipeline_tick_to_features(self, tick_data):
        """Test complete tick -> bars -> features pipeline."""
        # Tick -> Bars
        bars = aggregate_bars(
            tick_data,
            "dollar_bar",
            threshold=30000,
        )
        
        # Bars -> Features (with all features)
        config = UnifiedFeatureConfig(
            windows=[5, 10, 20],
            include_returns=True,
            include_volatility=True,
            include_momentum=True,
            include_volume=True,
            include_microstructure=True,
            include_alphas=False,
            include_technical=False,
        )
        calculator = UnifiedFeatureCalculator(config)
        features = calculator.calculate(bars)
        
        # Validate features
        assert len(features) == len(bars)
        feature_cols = [c for c in features.columns if c not in bars.columns]
        assert len(feature_cols) > 20  # Should have many features


# =============================================================================
# E2E Test: Cross-Sectional Backtest
# =============================================================================

class TestCrossSectionalBacktest:
    """Test cross-sectional backtesting workflow."""

    def test_full_cross_sectional_workflow(self, multi_asset_bars):
        """Test complete cross-sectional backtest workflow."""
        # Step 1: Calculate features per asset
        features = []
        for asset in multi_asset_bars["asset"].unique():
            asset_data = multi_asset_bars[multi_asset_bars["asset"] == asset].copy()
            asset_features = calculate_features(
                asset_data,
                windows=[5, 10, 20],
                include_alphas=False,
                include_technical=False,
                drop_na=False,
            )
            features.append(asset_features)
        
        feature_df = pd.concat(features, ignore_index=True)
        
        # Step 2: Prepare data for backtest (pivot to wide format)
        # Use timestamp as index for cross-sectional
        feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"])
        
        # Create price panel
        price_pivot = feature_df.pivot(
            index="timestamp",
            columns="asset",
            values="close",
        )
        
        # Create return panel for ranking
        return_pivot = feature_df.pivot(
            index="timestamp",
            columns="asset",
            values="return_20",
        )
        
        # Combine into multi-index DataFrame
        combined = pd.concat({
            "close": price_pivot,
            "return_20": return_pivot,
        }, axis=1)
        combined.columns = [f"{col[1]}_{col[0]}" for col in combined.columns]
        
        # Reshape to format expected by CrossSectionalEngine
        backtest_data = []
        for ts in price_pivot.index:
            for asset in price_pivot.columns:
                if pd.notna(price_pivot.loc[ts, asset]):
                    backtest_data.append({
                        "timestamp": ts,
                        "asset": asset,
                        "close": price_pivot.loc[ts, asset],
                        "return_20": return_pivot.loc[ts, asset] if pd.notna(return_pivot.loc[ts, asset]) else 0,
                    })
        
        backtest_df = pd.DataFrame(backtest_data)
        backtest_df = backtest_df.set_index(["timestamp", "asset"])
        
        # Step 3: Run backtest
        config = BacktestConfig(
            mode=BacktestMode.CROSS_SECTIONAL,
            initial_capital=1_000_000,
            start_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            rebalance_frequency="W",
            warmup_periods=30,
            risk_config=RiskConfigModel(
                max_position_size=0.25,
                max_leverage=1.0,
            ),
            cost_config=CostConfigModel(
                commission_rate=0.001,
                slippage_rate=0.0005,
            ),
        )
        
        strategy = MomentumStrategy(
            lookback_col="return_20",
            top_n_long=2,
            top_n_short=2,
        )
        
        engine = CrossSectionalEngine(config, strategy)
        result = engine.run(backtest_df)
        
        # Validate results
        assert result is not None
        assert result.initial_capital == 1_000_000
        assert result.final_capital > 0
        assert len(result.nav_history) > 0

    def test_momentum_vs_mean_reversion(self, multi_asset_bars):
        """Compare momentum and mean reversion strategies."""
        # Prepare data
        feature_df = multi_asset_bars.copy()
        feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"])
        
        # Calculate return_20 for all assets
        for asset in feature_df["asset"].unique():
            mask = feature_df["asset"] == asset
            feature_df.loc[mask, "return_20"] = feature_df.loc[mask, "close"].pct_change(20)
        
        backtest_df = feature_df.set_index(["timestamp", "asset"])
        
        config = BacktestConfig(
            mode=BacktestMode.CROSS_SECTIONAL,
            initial_capital=1_000_000,
            start_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 3, 20, tzinfo=timezone.utc),
            rebalance_frequency="W",
            warmup_periods=25,
        )
        
        # Test momentum strategy
        momentum_strategy = MomentumStrategy(lookback_col="return_20", top_n_long=2, top_n_short=2)
        momentum_engine = CrossSectionalEngine(config, momentum_strategy)
        momentum_result = momentum_engine.run(backtest_df)
        
        # Test mean reversion strategy
        mr_strategy = MeanReversionStrategy(lookback_col="return_20", top_n_long=2, top_n_short=2)
        mr_engine = CrossSectionalEngine(config, mr_strategy)
        mr_result = mr_engine.run(backtest_df)
        
        # Both should produce valid results
        assert momentum_result.final_capital > 0
        assert mr_result.final_capital > 0
        assert len(momentum_result.nav_history) > 0
        assert len(mr_result.nav_history) > 0


# =============================================================================
# E2E Test: Time-Series Backtest
# =============================================================================

class TestTimeSeriesBacktest:
    """Test time-series backtesting workflow."""

    def test_single_asset_time_series(self, tick_data):
        """Test single-asset time-series backtest."""
        # Step 1: Convert ticks to bars
        bars = build_dollar_bars(tick_data, dollar_threshold=40000)
        
        # Step 2: Calculate features
        features = calculate_features(
            bars,
            windows=[5, 10, 20],
            include_alphas=False,
            include_technical=False,
        )
        
        # Step 3: Run backtest
        config = BacktestConfig(
            mode=BacktestMode.TIME_SERIES,
            initial_capital=100_000,
            risk_config=RiskConfigModel(
                max_position_size=0.5,
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
            ),
            cost_config=CostConfigModel(
                commission_rate=0.001,
            ),
        )
        
        from crypto_data_engine.services.back_test.engine.time_series import (
            MomentumTimeSeriesStrategy,
        )
        
        strategy = MomentumTimeSeriesStrategy(
            momentum_column="momentum_20",
            long_threshold=0.02,
            short_threshold=-0.02,
        )
        
        engine = TimeSeriesEngine(config, strategy)
        result = engine.run(features)
        
        # Validate
        assert result is not None
        assert result.final_capital > 0

    def test_multi_asset_non_aligned(self, multi_asset_bars):
        """Test multi-asset backtest with non-aligned timestamps."""
        # Prepare multi-asset data dict
        data_dict = {}
        for asset in multi_asset_bars["asset"].unique()[:3]:  # Use first 3 assets
            asset_data = multi_asset_bars[multi_asset_bars["asset"] == asset].copy()
            asset_data = asset_data.drop(columns=["asset"])
            
            # Add features
            features = calculate_features(
                asset_data,
                windows=[5, 10],
                include_alphas=False,
                include_technical=False,
            )
            
            # Simulate non-aligned timestamps by adding random offset
            np.random.seed(hash(asset) % 2**32)
            offset = np.random.randint(0, 60, len(features))
            features["timestamp"] = pd.to_datetime(features["timestamp"]) + pd.to_timedelta(offset, unit="m")
            
            data_dict[asset] = features
        
        # Run backtest
        config = BacktestConfig(
            mode=BacktestMode.MULTI_ASSET_TIME_SERIES,
            initial_capital=500_000,
            risk_config=RiskConfigModel(
                max_position_size=0.3,
            ),
        )
        
        from crypto_data_engine.services.back_test.engine.time_series import (
            MeanReversionTimeSeriesStrategy,
        )
        
        strategy = MeanReversionTimeSeriesStrategy(
            zscore_column="momentum_10",
            entry_zscore=1.5,
            exit_zscore=0.5,
        )
        
        engine = TimeSeriesEngine(config, strategy)
        result = engine.run(data_dict)
        
        assert result is not None
        assert result.final_capital > 0


# =============================================================================
# E2E Test: Signal Generation Integration
# =============================================================================

class TestSignalIntegration:
    """Test signal generation integration with backtest."""

    def test_factor_signal_generation(self, multi_asset_bars):
        """Test factor-based signal generation."""
        # Calculate features
        feature_df = multi_asset_bars.copy()
        for asset in feature_df["asset"].unique():
            mask = feature_df["asset"] == asset
            asset_data = feature_df.loc[mask].copy()
            features = calculate_features(
                asset_data,
                windows=[10, 20],
                include_alphas=False,
                drop_na=False,
            )
            feature_df.loc[mask, "return_20"] = features["return_20"].values
            feature_df.loc[mask, "volatility_20"] = features["volatility_20"].values
        
        # Create factor signal generator
        factor_config = FactorConfig(
            name="return_20",
            weight=1.0,
            normalize=True,
        )
        
        signal_gen = FactorSignalGenerator(
            factors=[factor_config],
            long_threshold=0.6,
            short_threshold=0.4,
        )
        
        # Generate signals for a timestamp
        ts = feature_df["timestamp"].iloc[-1]
        cross_section = feature_df[feature_df["timestamp"] == ts]
        
        if len(cross_section) > 0:
            signal = signal_gen.generate(cross_section)
            
            assert signal is not None
            assert isinstance(signal, SignalOutput)

    def test_ensemble_signals(self, multi_asset_bars):
        """Test ensemble signal generation."""
        # Prepare data
        feature_df = multi_asset_bars.copy()
        for asset in feature_df["asset"].unique():
            mask = feature_df["asset"] == asset
            feature_df.loc[mask, "return_10"] = feature_df.loc[mask, "close"].pct_change(10)
            feature_df.loc[mask, "return_20"] = feature_df.loc[mask, "close"].pct_change(20)
        
        # Create multiple signal generators
        momentum_gen = RankSignalGenerator(
            factor_col="return_20",
            top_n_long=2,
            top_n_short=2,
        )
        
        short_momentum_gen = RankSignalGenerator(
            factor_col="return_10",
            top_n_long=2,
            top_n_short=2,
        )
        
        # Create ensemble
        ensemble = EnsembleSignalGenerator(
            generators=[
                GeneratorConfig(generator=momentum_gen, weight=0.6),
                GeneratorConfig(generator=short_momentum_gen, weight=0.4),
            ],
            method=EnsembleMethod.WEIGHTED_AVERAGE,
        )
        
        # Generate signal
        ts = feature_df["timestamp"].iloc[-1]
        cross_section = feature_df[feature_df["timestamp"] == ts].dropna(subset=["return_10", "return_20"])
        
        if len(cross_section) >= 2:
            signal = ensemble.generate(cross_section)
            assert signal is not None


# =============================================================================
# E2E Test: Asset Selection Integration
# =============================================================================

class TestAssetSelectionIntegration:
    """Test asset selection integration with backtest."""

    def test_volume_based_selection(self, multi_asset_bars):
        """Test volume-based asset selection."""
        # Create selector
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    min_value=50000,  # Minimum daily dollar volume
                ),
            ],
            max_assets=3,
        )
        
        # Select assets
        selected = selector.select(multi_asset_bars)
        
        assert len(selected) <= 3
        assert all(isinstance(a, str) for a in selected)

    def test_composite_selection(self, multi_asset_bars):
        """Test composite asset selection criteria."""
        # Add volatility column
        feature_df = multi_asset_bars.copy()
        for asset in feature_df["asset"].unique():
            mask = feature_df["asset"] == asset
            returns = feature_df.loc[mask, "close"].pct_change()
            feature_df.loc[mask, "volatility"] = returns.rolling(20).std()
        
        # Create selector with multiple filters
        selector = AssetSelector(
            filters=[
                SelectionFilter(
                    criteria=SelectionCriteria.DOLLAR_VOLUME,
                    min_value=10000,
                ),
                SelectionFilter(
                    criteria=SelectionCriteria.VOLATILITY,
                    min_value=0.01,
                    max_value=0.10,
                ),
            ],
            max_assets=4,
        )
        
        selected = selector.select(feature_df)
        assert len(selected) <= 4


# =============================================================================
# E2E Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory function integration."""

    def test_create_engine_cross_sectional(self, multi_asset_bars):
        """Test creating cross-sectional engine via factory."""
        config = BacktestConfig(
            mode=BacktestMode.CROSS_SECTIONAL,
            initial_capital=1_000_000,
            start_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 3, 1, tzinfo=timezone.utc),
            rebalance_frequency="W",
            warmup_periods=25,
        )
        
        strategy = create_strategy("momentum", lookback_col="return_20", top_n_long=2)
        engine = create_backtest_engine(config, strategy)
        
        assert isinstance(engine, CrossSectionalEngine)

    def test_create_engine_time_series(self, tick_data):
        """Test creating time-series engine via factory."""
        bars = build_dollar_bars(tick_data, dollar_threshold=30000)
        
        config = BacktestConfig(
            mode=BacktestMode.TIME_SERIES,
            initial_capital=100_000,
        )
        
        strategy = create_strategy("momentum_ts", momentum_column="momentum_20", long_threshold=0.02)
        engine = create_backtest_engine(config, strategy)
        
        assert isinstance(engine, TimeSeriesEngine)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
