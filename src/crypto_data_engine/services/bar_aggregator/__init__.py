"""
Bar aggregation module.

Provides tools for converting tick data into various bar types:
- Time bars (fixed time intervals)
- Tick bars (fixed number of ticks)
- Volume bars (fixed volume)
- Dollar bars (fixed dollar volume)

Usage:
    # Simple API
    from crypto_data_engine.services.bar_aggregator import aggregate_bars
    bars = aggregate_bars(tick_data, "dollar_bar", 1_000_000)
    
    # Convenience functions
    from crypto_data_engine.services.bar_aggregator import build_dollar_bars
    bars = build_dollar_bars(tick_data, dollar_threshold=1_000_000)
"""
from .bar_types import (
    BarType, BarConfig, BaseBarBuilder,
    TimeBarBuilder, TickBarBuilder, VolumeBarBuilder, DollarBarBuilder,
    get_bar_builder, build_bars
)
from .fast_aggregator import (
    FastBarAggregator, AggregationResult, StreamingAggregator, NUMBA_AVAILABLE
)
from .unified import (
    aggregate_bars,
    create_streaming_aggregator,
    build_time_bars,
    build_tick_bars,
    build_volume_bars,
    build_dollar_bars,
    benchmark_aggregation,
)
from .feature_calculator import (
    FeatureConfig, RollingFeatureCalculator, CrossSectionalFeatureCalculator,
    FeaturePipeline, calculate_weekly_returns, calculate_monthly_turnover
)
from .tick_feature_enricher import (
    TickFeatureEnricher, TickFeatureEnricherConfig, TICK_FEATURE_COLUMNS,
    enrich_file_pair_worker,
)

__all__ = [
    # Unified API (recommended)
    "aggregate_bars",
    "create_streaming_aggregator",
    "build_time_bars",
    "build_tick_bars",
    "build_volume_bars",
    "build_dollar_bars",
    "benchmark_aggregation",
    # Bar types
    "BarType",
    "BarConfig",
    "BaseBarBuilder",
    "TimeBarBuilder",
    "TickBarBuilder",
    "VolumeBarBuilder",
    "DollarBarBuilder",
    "get_bar_builder",
    "build_bars",
    # Fast aggregator
    "FastBarAggregator",
    "AggregationResult",
    "StreamingAggregator",
    "NUMBA_AVAILABLE",
    # Feature calculator
    "FeatureConfig",
    "RollingFeatureCalculator",
    "CrossSectionalFeatureCalculator",
    "FeaturePipeline",
    "calculate_weekly_returns",
    "calculate_monthly_turnover",
    # Tick feature enricher
    "TickFeatureEnricher",
    "TickFeatureEnricherConfig",
    "TICK_FEATURE_COLUMNS",
    "enrich_file_pair_worker",
]