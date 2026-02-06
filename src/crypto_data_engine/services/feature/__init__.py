"""
Feature calculation module.

Provides:
- UnifiedFeatureCalculator: All-in-one feature calculator
- Alpha factors (from Factor.py)
- Rolling and cross-sectional features

Usage:
    from crypto_data_engine.services.feature import calculate_features
    features = calculate_features(bar_data)
"""
from crypto_data_engine.services.feature.unified_features import (
    UnifiedFeatureConfig,
    UnifiedFeatureCalculator,
    calculate_features,
    calculate_features_multi_asset,
    select_features_by_correlation,
    get_feature_importance,
)

__all__ = [
    # Main API
    "calculate_features",
    "calculate_features_multi_asset",
    # Configuration
    "UnifiedFeatureConfig",
    "UnifiedFeatureCalculator",
    # Utilities
    "select_features_by_correlation",
    "get_feature_importance",
]
