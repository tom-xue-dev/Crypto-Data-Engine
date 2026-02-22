"""
Feature calculation module.

Provides:
- UnifiedFeatureCalculator: All-in-one feature calculator
- OrderFlowFactorCalculator: Order flow based factors (OFI, SmartFlow, VPIN, etc.)
- TickMicrostructureConfig / extract_daily_features: Tick-level microstructure factors
- Alpha factors (from Factor.py)
- Rolling and cross-sectional features

Usage:
    from crypto_data_engine.services.feature import calculate_features
    features = calculate_features(bar_data)

    # Order flow factors
    from crypto_data_engine.services.feature import calculate_order_flow_factors
    factors = calculate_order_flow_factors(dollar_bar_data)

    # Tick microstructure factors
    from crypto_data_engine.services.feature import extract_daily_features
    feats = extract_daily_features(tick_day_df)
"""
from crypto_data_engine.services.feature.unified_features import (
    UnifiedFeatureConfig,
    UnifiedFeatureCalculator,
    calculate_features,
    calculate_features_multi_asset,
    select_features_by_correlation,
    get_feature_importance,
)
from crypto_data_engine.services.feature.order_flow_factors import (
    OrderFlowFactorConfig,
    OrderFlowFactorCalculator,
    calculate_order_flow_factors,
    calculate_sweep_events,
    aggregate_sweeps_to_bars,
)
from crypto_data_engine.services.feature.tick_microstructure_factors import (
    TickMicrostructureConfig,
    compute_vpin,
    compute_toxicity,
    compute_kyle_lambda,
    compute_burstiness,
    compute_jump_ratio,
    compute_whale_metrics,
    extract_daily_features,
    process_tick_file,
)

__all__ = [
    # Main API
    "calculate_features",
    "calculate_features_multi_asset",
    # Configuration
    "UnifiedFeatureConfig",
    "UnifiedFeatureCalculator",
    # Order Flow Factors
    "OrderFlowFactorConfig",
    "OrderFlowFactorCalculator",
    "calculate_order_flow_factors",
    "calculate_sweep_events",
    "aggregate_sweeps_to_bars",
    # Tick Microstructure Factors
    "TickMicrostructureConfig",
    "compute_vpin",
    "compute_toxicity",
    "compute_kyle_lambda",
    "compute_burstiness",
    "compute_jump_ratio",
    "compute_whale_metrics",
    "extract_daily_features",
    "process_tick_file",
    # Utilities
    "select_features_by_correlation",
    "get_feature_importance",
]
