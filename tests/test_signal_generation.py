"""
Unit tests for signal generation module.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.core.base import SignalType
from crypto_data_engine.services.signal_generation import (
    BaseSignalGenerator,
    SignalOutput,
    FactorSignalGenerator,
    FactorConfig,
    RankSignalGenerator,
    ThresholdSignalGenerator,
    RuleSignalGenerator,
    RuleCondition,
    ComparisonOperator,
    EnsembleSignalGenerator,
    EnsembleMethod,
)
from crypto_data_engine.services.signal_generation.ensemble import GeneratorConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_cross_section():
    """Create sample cross-sectional data."""
    np.random.seed(42)
    assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
    
    data = pd.DataFrame({
        "close": [50000, 3000, 300, 100, 0.5],
        "return_5": [0.05, -0.02, 0.03, -0.05, 0.01],
        "return_20": [0.10, -0.05, 0.08, -0.10, 0.02],
        "momentum": [0.12, -0.03, 0.05, -0.08, 0.01],
        "rsi": [65, 30, 55, 25, 50],
        "zscore": [1.5, -2.0, 0.5, -1.5, 0.0],
        "sma_10": [49000, 3100, 295, 105, 0.48],
        "sma_50": [48000, 3200, 290, 110, 0.45],
    }, index=assets)
    
    return data


@pytest.fixture
def sample_timeseries_data():
    """Create sample time-series data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    
    data = pd.DataFrame({
        "close": 100 * np.cumprod(1 + np.random.randn(50) * 0.02),
        "volume": np.random.randint(1000, 10000, 50),
        "rsi": np.random.uniform(20, 80, 50),
    }, index=dates)
    
    data["sma_10"] = data["close"].rolling(10).mean()
    data["sma_50"] = data["close"].rolling(50).mean()
    
    return data


# =============================================================================
# SignalOutput Tests
# =============================================================================

class TestSignalOutput:
    """Tests for SignalOutput class."""

    def test_from_weights(self):
        """Test creating SignalOutput from weights."""
        weights = {"BTC": 0.3, "ETH": -0.2, "BNB": 0.0}
        timestamp = datetime(2024, 1, 1)
        
        output = SignalOutput.from_weights(timestamp, weights)
        
        assert output.signals["BTC"] == SignalType.BUY
        assert output.signals["ETH"] == SignalType.SELL
        assert output.signals["BNB"] == SignalType.HOLD
        
        assert output.strengths["BTC"] > 0
        assert output.strengths["ETH"] < 0

    def test_from_strengths(self):
        """Test creating SignalOutput from strengths."""
        strengths = {"BTC": 0.5, "ETH": -0.3, "BNB": 0.05}
        timestamp = datetime(2024, 1, 1)
        
        output = SignalOutput.from_strengths(
            timestamp, strengths,
            long_threshold=0.1,
            short_threshold=-0.1,
        )
        
        assert output.signals["BTC"] == SignalType.BUY
        assert output.signals["ETH"] == SignalType.SELL
        assert output.signals["BNB"] == SignalType.HOLD

    def test_to_dict(self):
        """Test serialization to dictionary."""
        output = SignalOutput(
            timestamp=datetime(2024, 1, 1),
            signals={"BTC": SignalType.BUY},
            strengths={"BTC": 0.5},
            weights={"BTC": 0.3},
        )
        
        d = output.to_dict()
        
        assert "timestamp" in d
        assert d["signals"]["BTC"] == "BUY"
        assert d["strengths"]["BTC"] == 0.5


# =============================================================================
# FactorSignalGenerator Tests
# =============================================================================

class TestFactorSignalGenerator:
    """Tests for FactorSignalGenerator."""

    def test_basic_signal_generation(self, sample_cross_section):
        """Test basic factor signal generation."""
        factors = [
            FactorConfig(name="momentum", direction=1, weight=1.0),
        ]
        
        generator = FactorSignalGenerator(
            factors=factors,
            long_threshold=0.5,
            short_threshold=-0.5,
        )
        
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        assert len(output.strengths) > 0
        assert all(isinstance(s, SignalType) for s in output.signals.values())

    def test_multi_factor(self, sample_cross_section):
        """Test multi-factor signal generation."""
        factors = [
            FactorConfig(name="momentum", direction=1, weight=2.0),
            FactorConfig(name="return_20", direction=1, weight=1.0),
        ]
        
        generator = FactorSignalGenerator(
            factors=factors,
            top_n_long=2,
            top_n_short=2,
        )
        
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # Should have 4 positions
        non_zero_weights = [w for w in output.weights.values() if w != 0]
        assert len(non_zero_weights) == 4

    def test_normalization(self, sample_cross_section):
        """Test z-score normalization."""
        factors = [
            FactorConfig(name="momentum", normalize=True),
        ]
        
        generator = FactorSignalGenerator(factors=factors)
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # Normalized strengths should be roughly centered around 0
        strengths = list(output.strengths.values())
        assert abs(np.mean(strengths)) < 0.5


class TestRankSignalGenerator:
    """Tests for RankSignalGenerator."""

    def test_basic_ranking(self, sample_cross_section):
        """Test basic rank-based signal generation."""
        generator = RankSignalGenerator(
            factor_col="momentum",
            top_n_long=2,
            top_n_short=2,
            ascending=False,
        )
        
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # Check we have correct number of signals
        long_signals = [a for a, s in output.signals.items() if s == SignalType.BUY]
        short_signals = [a for a, s in output.signals.items() if s == SignalType.SELL]
        
        assert len(long_signals) == 2
        assert len(short_signals) == 2

    def test_equal_weights(self, sample_cross_section):
        """Test equal weight distribution."""
        generator = RankSignalGenerator(
            factor_col="momentum",
            top_n_long=2,
            top_n_short=2,
            equal_weight=True,
        )
        
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # Long weights should be equal
        long_weights = [w for w in output.weights.values() if w > 0]
        assert len(set([round(w, 4) for w in long_weights])) == 1


class TestThresholdSignalGenerator:
    """Tests for ThresholdSignalGenerator."""

    def test_threshold_signals(self, sample_cross_section):
        """Test threshold-based signals.
        
        ThresholdSignalGenerator logic:
        - value > long_threshold → BUY (momentum: high is good)
        - value < short_threshold → SELL (momentum: low is bad)
        """
        generator = ThresholdSignalGenerator(
            factor_col="zscore",
            long_threshold=1.0,   # Above 1.0 → BUY
            short_threshold=-1.0, # Below -1.0 → SELL
            use_zscore=False,
        )
        
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # BTCUSDT has zscore=1.5 (above 1.0), should be BUY
        assert output.signals.get("BTCUSDT") == SignalType.BUY
        
        # ETHUSDT has zscore=-2.0 (below -1.0), should be SELL
        assert output.signals.get("ETHUSDT") == SignalType.SELL
        
        # SOLUSDT has zscore=-1.5 (below -1.0), should be SELL
        assert output.signals.get("SOLUSDT") == SignalType.SELL


# =============================================================================
# RuleSignalGenerator Tests
# =============================================================================

class TestRuleCondition:
    """Tests for RuleCondition."""

    def test_greater_than(self):
        """Test greater than condition."""
        condition = RuleCondition("rsi", ComparisonOperator.GREATER, 70)
        
        row = pd.Series({"rsi": 75})
        assert condition.evaluate(row) == True
        
        row = pd.Series({"rsi": 65})
        assert condition.evaluate(row) == False

    def test_between(self):
        """Test between condition."""
        condition = RuleCondition("rsi", ComparisonOperator.BETWEEN, (30, 70))
        
        row = pd.Series({"rsi": 50})
        assert condition.evaluate(row) == True
        
        row = pd.Series({"rsi": 80})
        assert condition.evaluate(row) == False

    def test_column_comparison(self):
        """Test comparing two columns."""
        condition = RuleCondition("sma_10", ComparisonOperator.GREATER, "sma_50")
        
        row = pd.Series({"sma_10": 100, "sma_50": 95})
        assert condition.evaluate(row) == True
        
        row = pd.Series({"sma_10": 90, "sma_50": 95})
        assert condition.evaluate(row) == False

    def test_cross_above(self):
        """Test cross above condition."""
        condition = RuleCondition("sma_10", ComparisonOperator.CROSS_ABOVE, "sma_50")
        
        prev_row = pd.Series({"sma_10": 95, "sma_50": 100})
        curr_row = pd.Series({"sma_10": 105, "sma_50": 100})
        
        assert condition.evaluate(curr_row, prev_row) == True
        
        prev_row = pd.Series({"sma_10": 105, "sma_50": 100})
        curr_row = pd.Series({"sma_10": 110, "sma_50": 100})
        
        assert condition.evaluate(curr_row, prev_row) == False


class TestRuleSignalGenerator:
    """Tests for RuleSignalGenerator."""

    def test_rsi_strategy(self, sample_cross_section):
        """Test RSI-based strategy."""
        generator = RuleSignalGenerator(
            long_conditions=[
                RuleCondition("rsi", ComparisonOperator.LESS, 30),
            ],
            short_conditions=[
                RuleCondition("rsi", ComparisonOperator.GREATER, 60),
            ],
        )
        
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # SOLUSDT has rsi=25, should be long
        assert output.signals.get("SOLUSDT") == SignalType.BUY
        
        # BTCUSDT has rsi=65, should be short
        assert output.signals.get("BTCUSDT") == SignalType.SELL

    def test_multiple_conditions_and(self, sample_cross_section):
        """Test AND logic with multiple conditions."""
        generator = RuleSignalGenerator(
            long_conditions=[
                RuleCondition("rsi", ComparisonOperator.LESS, 40),
                RuleCondition("momentum", ComparisonOperator.LESS, 0),
            ],
            require_all_long=True,
        )
        
        output = generator.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # Only assets meeting BOTH conditions should be long
        for asset, signal in output.signals.items():
            if signal == SignalType.BUY:
                assert sample_cross_section.loc[asset, "rsi"] < 40
                assert sample_cross_section.loc[asset, "momentum"] < 0


# =============================================================================
# EnsembleSignalGenerator Tests
# =============================================================================

class TestEnsembleSignalGenerator:
    """Tests for EnsembleSignalGenerator."""

    def test_weighted_average(self, sample_cross_section):
        """Test weighted average ensemble."""
        gen1 = RankSignalGenerator("momentum", top_n_long=2, top_n_short=2)
        gen2 = RankSignalGenerator("return_20", top_n_long=2, top_n_short=2)
        
        ensemble = EnsembleSignalGenerator(
            generators=[
                GeneratorConfig(gen1, weight=2.0),
                GeneratorConfig(gen2, weight=1.0),
            ],
            method=EnsembleMethod.WEIGHTED_AVERAGE,
        )
        
        output = ensemble.generate(sample_cross_section, datetime(2024, 1, 1))
        
        assert len(output.strengths) > 0

    def test_voting(self, sample_cross_section):
        """Test voting ensemble."""
        gen1 = RankSignalGenerator("momentum", top_n_long=3, top_n_short=0)
        gen2 = RankSignalGenerator("return_20", top_n_long=3, top_n_short=0)
        gen3 = ThresholdSignalGenerator("zscore", long_threshold=-0.5, short_threshold=0.5)
        
        ensemble = EnsembleSignalGenerator(
            generators=[
                GeneratorConfig(gen1),
                GeneratorConfig(gen2),
                GeneratorConfig(gen3),
            ],
            method=EnsembleMethod.VOTING,
            min_agreement=0.5,
        )
        
        output = ensemble.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # Should have some signals
        assert len(output.signals) > 0

    def test_unanimous(self, sample_cross_section):
        """Test unanimous ensemble (conservative)."""
        # Both generators agree on BTCUSDT being best
        gen1 = RankSignalGenerator("momentum", top_n_long=1, top_n_short=0)
        gen2 = RankSignalGenerator("return_20", top_n_long=1, top_n_short=0)
        
        ensemble = EnsembleSignalGenerator(
            generators=[
                GeneratorConfig(gen1),
                GeneratorConfig(gen2),
            ],
            method=EnsembleMethod.UNANIMOUS,
        )
        
        output = ensemble.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # Unanimous requires all to agree, so might have fewer signals
        assert isinstance(output, SignalOutput)


# =============================================================================
# Integration Tests
# =============================================================================

class TestSignalGenerationIntegration:
    """Integration tests for signal generation workflow."""

    def test_full_workflow(self, sample_cross_section):
        """Test complete signal generation workflow."""
        # 1. Create factor-based generator
        factor_gen = FactorSignalGenerator(
            factors=[
                FactorConfig("momentum", direction=1, weight=1.0),
                FactorConfig("return_20", direction=1, weight=0.5),
            ],
            top_n_long=2,
            top_n_short=2,
        )
        
        # 2. Create rule-based generator
        rule_gen = RuleSignalGenerator(
            long_conditions=[
                RuleCondition("rsi", ComparisonOperator.LESS, 40),
            ],
            short_conditions=[
                RuleCondition("rsi", ComparisonOperator.GREATER, 60),
            ],
        )
        
        # 3. Create ensemble
        ensemble = EnsembleSignalGenerator(
            generators=[
                GeneratorConfig(factor_gen, weight=2.0),
                GeneratorConfig(rule_gen, weight=1.0),
            ],
            method=EnsembleMethod.WEIGHTED_AVERAGE,
        )
        
        # 4. Generate signals
        output = ensemble.generate(sample_cross_section, datetime(2024, 1, 1))
        
        # 5. Verify output structure
        assert output.timestamp == datetime(2024, 1, 1)
        assert len(output.signals) > 0
        assert len(output.strengths) > 0
        assert len(output.weights) > 0
        
        # 6. Convert to DataFrame
        outputs = [output]
        df = factor_gen.to_dataframe(outputs)
        
        assert "timestamp" in df.columns
        assert "asset" in df.columns
        assert "signal" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
