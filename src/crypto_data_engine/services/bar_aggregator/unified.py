"""
Unified bar aggregation interface.

Provides a single entry point for all bar aggregation operations,
combining the functionality of multiple implementations with automatic
optimization selection.

Usage:
    from crypto_data_engine.services.bar_aggregator import aggregate_bars
    
    # Simple usage
    bars = aggregate_bars(tick_data, "dollar_bar", threshold=1_000_000)
    
    # With configuration
    bars = aggregate_bars(
        tick_data,
        bar_type="time_bar",
        threshold="5min",
        use_numba=True,
        include_advanced=True,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .bar_types import (
    BarType,
    BarConfig,
    BaseBarBuilder,
    get_bar_builder,
    build_bars as _build_bars_simple,
)
from .fast_aggregator import (
    FastBarAggregator,
    StreamingAggregator,
    AggregationResult,
    NUMBA_AVAILABLE,
)


def aggregate_bars(
    data: Union[pd.DataFrame, str, Path, List[str]],
    bar_type: Union[BarType, str],
    threshold: Union[int, float, str],
    *,
    use_numba: bool = True,
    use_multiprocess: bool = True,
    n_workers: int = 4,
    include_advanced: bool = True,
    symbol: str = "UNKNOWN",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Union[pd.DataFrame, Dict[str, AggregationResult]]:
    """
    Unified bar aggregation function.
    
    Automatically selects the best implementation based on:
    - Data type and bar type
    - Available optimizations (Numba)
    - Dataset size
    
    Args:
        data: Input data - DataFrame, file path, or list of file paths
        bar_type: Type of bars to build
        threshold: Bar threshold (count for tick, volume for volume bar, etc.)
        use_numba: Use Numba acceleration if available (default True)
        use_multiprocess: Use multiprocessing for batch files (default True)
        n_workers: Number of worker processes (default 4)
        include_advanced: Include advanced microstructure features (default True)
        symbol: Symbol name for results
        progress_callback: Callback function(completed, total) for progress
        
    Returns:
        DataFrame of bars (for single input) or Dict of results (for batch)
        
    Examples:
        # From DataFrame
        bars = aggregate_bars(tick_df, "dollar_bar", 1_000_000)
        
        # From file
        bars = aggregate_bars("data/ticks.parquet", "time_bar", "5min")
        
        # Batch processing
        results = aggregate_bars(
            ["file1.parquet", "file2.parquet"],
            "volume_bar",
            100_000,
            n_workers=8,
        )
    """
    # Normalize bar type
    if isinstance(bar_type, str):
        bar_type = BarType(bar_type)
    
    # Handle different input types
    if isinstance(data, (str, Path)):
        # Single file
        return _aggregate_file(
            data, bar_type, threshold,
            use_numba=use_numba,
            include_advanced=include_advanced,
            symbol=symbol,
        )
    
    elif isinstance(data, list):
        # Batch files
        return _aggregate_batch(
            data, bar_type, threshold,
            use_numba=use_numba,
            use_multiprocess=use_multiprocess,
            n_workers=n_workers,
            include_advanced=include_advanced,
            progress_callback=progress_callback,
        )
    
    elif isinstance(data, pd.DataFrame):
        # DataFrame
        return _aggregate_dataframe(
            data, bar_type, threshold,
            use_numba=use_numba,
            include_advanced=include_advanced,
            symbol=symbol,
        )
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _aggregate_dataframe(
    data: pd.DataFrame,
    bar_type: BarType,
    threshold: Union[int, float, str],
    use_numba: bool = True,
    include_advanced: bool = True,
    symbol: str = "UNKNOWN",
) -> pd.DataFrame:
    """Aggregate a DataFrame of tick data."""
    from .tick_normalizer import normalize_tick_data

    # Normalize column names, timestamp units, and dtypes
    data = normalize_tick_data(data)

    # Determine best method – Numba now covers TIME_BAR as well
    should_use_numba = (
        use_numba
        and NUMBA_AVAILABLE
        and bar_type in (BarType.TIME_BAR, BarType.VOLUME_BAR, BarType.DOLLAR_BAR)
        and len(data) > 10000  # Only for larger datasets
    )
    
    if should_use_numba:
        aggregator = FastBarAggregator(
            use_numba=True,
            use_multiprocess=False,
            include_advanced=include_advanced,
        )
        result = aggregator.aggregate(data, bar_type, threshold, symbol)
        return result.bars
    else:
        # Use pandas-based builder
        config = BarConfig(
            bar_type=bar_type,
            threshold=threshold,
            include_advanced_features=include_advanced,
        )
        builder = get_bar_builder(config)
        return builder.build_bars(data)


def _aggregate_file(
    file_path: Union[str, Path],
    bar_type: BarType,
    threshold: Union[int, float, str],
    use_numba: bool = True,
    include_advanced: bool = True,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate a single file."""
    file_path = Path(file_path)
    
    if symbol is None:
        symbol = file_path.stem
    
    # Load data
    if file_path.suffix == ".parquet":
        data = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        data = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return _aggregate_dataframe(
        data, bar_type, threshold,
        use_numba=use_numba,
        include_advanced=include_advanced,
        symbol=symbol,
    )


def _aggregate_batch(
    file_paths: List[Union[str, Path]],
    bar_type: BarType,
    threshold: Union[int, float, str],
    use_numba: bool = True,
    use_multiprocess: bool = True,
    n_workers: int = 4,
    include_advanced: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, AggregationResult]:
    """Aggregate multiple files."""
    aggregator = FastBarAggregator(
        use_numba=use_numba and NUMBA_AVAILABLE,
        use_multiprocess=use_multiprocess,
        n_workers=n_workers,
        include_advanced=include_advanced,
    )
    
    return aggregator.aggregate_batch(
        file_paths, bar_type, threshold,
        progress_callback=progress_callback,
    )


def create_streaming_aggregator(
    bar_type: Union[BarType, str],
    threshold: Union[int, float, str],
    chunk_size: int = 1_000_000,
) -> StreamingAggregator:
    """
    Create a streaming aggregator for processing large datasets in chunks.
    
    Args:
        bar_type: Type of bars to build
        threshold: Bar threshold
        chunk_size: Size of chunks to process
        
    Returns:
        StreamingAggregator instance
        
    Examples:
        aggregator = create_streaming_aggregator("dollar_bar", 1_000_000)
        
        for chunk in pd.read_parquet(file, chunksize=1_000_000):
            bars = aggregator.process_chunk(chunk)
            # Process bars...
        
        final_bars = aggregator.finalize()
    """
    if isinstance(bar_type, str):
        bar_type = BarType(bar_type)
    
    return StreamingAggregator(
        bar_type=bar_type,
        threshold=threshold,
        chunk_size=chunk_size,
    )


# Convenience functions for specific bar types
def build_time_bars(
    data: Union[pd.DataFrame, str, Path],
    interval: str = "5min",
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Build time bars with specified interval.
    
    Args:
        data: Tick data (DataFrame or file path)
        interval: Time interval (e.g., "1min", "5min", "1h")
        include_advanced: Include advanced features
        
    Returns:
        DataFrame of time bars
    """
    return aggregate_bars(
        data,
        BarType.TIME_BAR,
        interval,
        include_advanced=include_advanced,
    )


def build_tick_bars(
    data: Union[pd.DataFrame, str, Path],
    n_ticks: int = 1000,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Build tick bars with specified tick count.
    
    Args:
        data: Tick data
        n_ticks: Number of ticks per bar
        include_advanced: Include advanced features
        
    Returns:
        DataFrame of tick bars
    """
    return aggregate_bars(
        data,
        BarType.TICK_BAR,
        n_ticks,
        include_advanced=include_advanced,
    )


def build_volume_bars(
    data: Union[pd.DataFrame, str, Path],
    volume_threshold: float = 100_000,
    use_numba: bool = True,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Build volume bars with specified volume threshold.
    
    Args:
        data: Tick data
        volume_threshold: Volume per bar
        use_numba: Use Numba acceleration
        include_advanced: Include advanced features
        
    Returns:
        DataFrame of volume bars
    """
    return aggregate_bars(
        data,
        BarType.VOLUME_BAR,
        volume_threshold,
        use_numba=use_numba,
        include_advanced=include_advanced,
    )


def build_dollar_bars(
    data: Union[pd.DataFrame, str, Path],
    dollar_threshold: float = 1_000_000,
    use_numba: bool = True,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Build dollar bars with specified dollar volume threshold.
    
    Args:
        data: Tick data
        dollar_threshold: Dollar volume per bar
        use_numba: Use Numba acceleration
        include_advanced: Include advanced features
        
    Returns:
        DataFrame of dollar bars
    """
    return aggregate_bars(
        data,
        BarType.DOLLAR_BAR,
        dollar_threshold,
        use_numba=use_numba,
        include_advanced=include_advanced,
    )


# =============================================================================
# Performance utilities
# =============================================================================

def benchmark_aggregation(
    data: pd.DataFrame,
    bar_type: Union[BarType, str],
    threshold: Union[int, float, str],
    n_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark different aggregation methods.
    
    Args:
        data: Tick data to aggregate
        bar_type: Bar type
        threshold: Threshold
        n_iterations: Number of iterations for timing
        
    Returns:
        Dict with timing results and recommendations
    """
    import time
    
    if isinstance(bar_type, str):
        bar_type = BarType(bar_type)
    
    results = {
        "data_size": len(data),
        "bar_type": bar_type.value,
        "threshold": threshold,
        "methods": {},
    }
    
    # Test pandas method
    pandas_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        config = BarConfig(bar_type, threshold, include_advanced_features=True)
        builder = get_bar_builder(config)
        bars = builder.build_bars(data)
        pandas_times.append(time.perf_counter() - start)
    
    results["methods"]["pandas"] = {
        "mean_time": np.mean(pandas_times),
        "std_time": np.std(pandas_times),
        "bar_count": len(bars),
    }
    
    # Test Numba method if applicable
    if NUMBA_AVAILABLE and bar_type in (BarType.VOLUME_BAR, BarType.DOLLAR_BAR):
        # Warm up JIT
        aggregator = FastBarAggregator(use_numba=True, include_advanced=True)
        _ = aggregator.aggregate(data, bar_type, threshold)
        
        numba_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result = aggregator.aggregate(data, bar_type, threshold)
            numba_times.append(time.perf_counter() - start)
        
        results["methods"]["numba"] = {
            "mean_time": np.mean(numba_times),
            "std_time": np.std(numba_times),
            "bar_count": result.bar_count,
        }
        
        speedup = results["methods"]["pandas"]["mean_time"] / results["methods"]["numba"]["mean_time"]
        results["numba_speedup"] = speedup
        results["recommendation"] = "numba" if speedup > 1.2 else "pandas"
    else:
        results["recommendation"] = "pandas"
        results["numba_available"] = NUMBA_AVAILABLE
    
    return results
