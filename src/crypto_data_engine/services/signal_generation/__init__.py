"""
Signal generation module for quantitative trading.

This module provides:
- BaseSignalGenerator: Abstract base class for signal generators
- FactorSignalGenerator: Generate signals from factor values
- RuleSignalGenerator: Generate signals from rule-based conditions
- EnsembleSignalGenerator: Combine multiple signal sources
- OrderFlowMomentumStrategy: Order flow based momentum strategy
- MeanReversionStrategy: Mean reversion with order flow confirmation
"""
from crypto_data_engine.services.signal_generation.base import (
    BaseSignalGenerator,
    SignalOutput,
)
from crypto_data_engine.services.signal_generation.factor_signal import (
    FactorSignalGenerator,
    FactorConfig,
    RankSignalGenerator,
    ThresholdSignalGenerator,
)
from crypto_data_engine.services.signal_generation.rule_signal import (
    RuleSignalGenerator,
    RuleCondition,
    ComparisonOperator,
)
from crypto_data_engine.services.signal_generation.ensemble import (
    EnsembleSignalGenerator,
    EnsembleMethod,
    GeneratorConfig,
)
from crypto_data_engine.services.signal_generation.order_flow_strategy import (
    SignalType as OrderFlowSignalType,
    ExitReason,
    PositionState,
    OrderFlowMomentumConfig,
    OrderFlowMomentumStrategy,
    MeanReversionConfig,
    MeanReversionStrategy,
    SignalSummary,
    summarize_signals,
)

__all__ = [
    # Base
    "BaseSignalGenerator",
    "SignalOutput",
    # Factor-based
    "FactorSignalGenerator",
    "FactorConfig",
    "RankSignalGenerator",
    "ThresholdSignalGenerator",
    # Rule-based
    "RuleSignalGenerator",
    "RuleCondition",
    "ComparisonOperator",
    # Ensemble
    "EnsembleSignalGenerator",
    "EnsembleMethod",
    "GeneratorConfig",
    # Order Flow Strategies
    "OrderFlowSignalType",
    "ExitReason",
    "PositionState",
    "OrderFlowMomentumConfig",
    "OrderFlowMomentumStrategy",
    "MeanReversionConfig",
    "MeanReversionStrategy",
    "SignalSummary",
    "summarize_signals",
]
