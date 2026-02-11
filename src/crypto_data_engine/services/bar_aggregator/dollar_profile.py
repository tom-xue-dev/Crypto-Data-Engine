"""
Daily dollar volume profiling for dynamic dollar bar thresholds.

Phase 1 of the dynamic dollar bar pipeline:
- Scans tick parquet files for each symbol
- Computes daily dollar volume (sum of price * quantity per day)
- Caches profiles as lightweight parquet files
- Provides rolling lookback threshold queries

Core formula:
    threshold = avg_daily_dollar_volume(past N days) / K
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 10
DEFAULT_BARS_PER_DAY = 50
DEFAULT_DISCARD_MONTHS = 1


def compute_daily_dollar_volume(tick_file: Path) -> pd.DataFrame:
    """Compute daily dollar volume from a single tick parquet file.

    Only reads the three columns needed for profile computation
    (timestamp, price, quantity) to minimize I/O. Handles both
    named-column and numeric-column (headerless CSV → parquet) formats.

    Returns:
        DataFrame with columns [date, daily_dollar_volume].
        May be empty if the file has no valid data.
    """
    try:
        raw_data = _read_profile_columns(tick_file)
    except Exception as error:
        logger.debug(f"Skipping {tick_file.name} for profile: {error}")
        return pd.DataFrame(columns=["date", "daily_dollar_volume"])

    if raw_data.empty:
        return pd.DataFrame(columns=["date", "daily_dollar_volume"])

    timestamp_col, price_col, quantity_col = raw_data.columns[0], raw_data.columns[1], raw_data.columns[2]

    # Detect timestamp unit and convert to milliseconds
    from crypto_data_engine.services.bar_aggregator.tick_normalizer import (
        _normalize_timestamps_to_ms,
    )
    timestamps_ms = _normalize_timestamps_to_ms(raw_data[timestamp_col])

    price = raw_data[price_col].astype(np.float64)
    quantity = raw_data[quantity_col].astype(np.float64)

    # Compute dollar volume per tick and group by day
    dollar_volume = price * quantity
    dates = pd.to_datetime(timestamps_ms, unit="ms", utc=True).dt.date

    daily_volume = (
        pd.DataFrame({"date": dates, "dollar_volume": dollar_volume})
        .groupby("date")["dollar_volume"]
        .sum()
        .reset_index()
        .rename(columns={"dollar_volume": "daily_dollar_volume"})
    )

    return daily_volume


def _read_profile_columns(tick_file: Path) -> pd.DataFrame:
    """Read only timestamp, price, quantity from a tick parquet file.

    Handles two common schemas:
    - Named columns: reads ['timestamp', 'price', 'quantity'] (with alias fallback)
    - Numeric/positional columns: reads columns [1, 2, 5] (Binance aggTrades order)

    Returns:
        DataFrame with exactly 3 columns in order [timestamp, price, quantity].
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(tick_file)
    schema_names = parquet_file.schema.names

    # Check if columns are named (semantic) or positional (numeric indices)
    named_candidates = {
        "timestamp": ["timestamp", "transact_time", "trade_time", "time", "ts", "T"],
        "price": ["price", "p"],
        "quantity": ["quantity", "qty", "q", "size", "amount"],
    }

    resolved = {}
    for standard_name, aliases in named_candidates.items():
        for alias in aliases:
            if alias in schema_names:
                resolved[standard_name] = alias
                break

    if len(resolved) == 3:
        # All three columns found by name — read only those
        columns_to_read = [resolved["timestamp"], resolved["price"], resolved["quantity"]]
        dataframe = pq.read_table(tick_file, columns=columns_to_read).to_pandas()
        dataframe.columns = ["timestamp", "price", "quantity"]
        return dataframe

    # Fallback: positional columns (Binance aggTrades: col 5=timestamp, 1=price, 2=quantity)
    total_columns = len(schema_names)
    if total_columns >= 6:
        # Read by positional index via column indices
        dataframe = pq.read_table(tick_file, columns=[schema_names[5], schema_names[1], schema_names[2]]).to_pandas()
        dataframe.columns = ["timestamp", "price", "quantity"]
        return dataframe

    # Last resort: read all and normalize
    from crypto_data_engine.services.bar_aggregator.tick_normalizer import (
        normalize_tick_data,
    )
    raw_data = pd.read_parquet(tick_file)
    normalized = normalize_tick_data(raw_data, source_hint=tick_file.name)
    return normalized[["timestamp", "price", "quantity"]]


def build_symbol_profile(
    symbol_dir: Path,
    cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build or load cached daily dollar volume profile for a symbol.

    Scans all tick parquet files in the symbol directory, computes daily
    dollar volumes, and optionally caches the result.

    Args:
        symbol_dir: Directory containing tick parquet files for one symbol.
        cache_dir: Directory to cache profile parquet files. None = no caching.
        force_rebuild: Rebuild profile even if cached version exists.

    Returns:
        DataFrame with columns [date, daily_dollar_volume], sorted by date.
    """
    symbol = symbol_dir.name

    # Check cache
    if cache_dir and not force_rebuild:
        cache_path = cache_dir / f"{symbol}_daily_profile.parquet"
        if cache_path.exists():
            logger.debug(f"Loading cached profile for {symbol}")
            return pd.read_parquet(cache_path)

    tick_files = sorted(symbol_dir.glob("*.parquet"))
    if not tick_files:
        logger.debug(f"No tick files found for {symbol}")
        return pd.DataFrame(columns=["date", "daily_dollar_volume"])

    daily_frames = []
    for tick_file in tick_files:
        daily_volume = compute_daily_dollar_volume(tick_file)
        if not daily_volume.empty:
            daily_frames.append(daily_volume)

    if not daily_frames:
        return pd.DataFrame(columns=["date", "daily_dollar_volume"])

    profile = pd.concat(daily_frames, ignore_index=True)

    # Aggregate in case multiple files cover the same day (unlikely but safe)
    profile = (
        profile.groupby("date")["daily_dollar_volume"]
        .sum()
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Cache if requested
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{symbol}_daily_profile.parquet"
        profile.to_parquet(cache_path, index=False)
        logger.debug(f"Cached profile for {symbol}: {len(profile)} days -> {cache_path}")

    return profile


def get_dynamic_threshold(
    profile: pd.DataFrame,
    target_date: datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    bars_per_day: int = DEFAULT_BARS_PER_DAY,
    use_ema: bool = False,
) -> Optional[float]:
    """Query dynamic dollar bar threshold for a given date.

    Computes:
        SMA mode: threshold = mean(daily_dollar_volume over past N days) / K
        EMA mode: threshold = EMA(daily_dollar_volume, span=N).last / K

    EMA gives more weight to recent days, adapting faster to volume regime changes.

    Args:
        profile: Daily dollar volume profile (from build_symbol_profile).
        target_date: The date to compute threshold for (uses preceding N days).
        lookback_days: Number of past days to average (N).
        bars_per_day: Target number of bars per day (K).
        use_ema: If True, use Exponential Moving Average instead of simple mean.

    Returns:
        Threshold value (float), or None if insufficient history.
    """
    if profile.empty or bars_per_day <= 0:
        return None

    # Ensure date column is comparable
    if isinstance(target_date, datetime):
        target_date_val = target_date.date() if hasattr(target_date, "date") else target_date
    else:
        target_date_val = target_date

    # Filter to days strictly before the target date
    preceding_data = profile[profile["date"] < target_date_val]

    if len(preceding_data) < lookback_days:
        return None

    if use_ema:
        # EMA over all preceding data with span=N (most recent value is the estimate)
        ema_series = preceding_data["daily_dollar_volume"].ewm(span=lookback_days).mean()
        average_daily_volume = ema_series.iloc[-1]
    else:
        # Simple mean of the most recent N days
        recent_days = preceding_data.tail(lookback_days)
        average_daily_volume = recent_days["daily_dollar_volume"].mean()

    if average_daily_volume <= 0 or np.isnan(average_daily_volume):
        return None

    threshold = average_daily_volume / bars_per_day
    return threshold


def get_first_valid_date(
    profile: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    discard_months: int = DEFAULT_DISCARD_MONTHS,
) -> Optional[datetime]:
    """Determine the first date with enough history for dynamic thresholds.

    Takes the later of:
    - The date after the first `lookback_days` days of data
    - The date after discarding the first `discard_months` months

    Returns:
        First valid date, or None if not enough data.
    """
    if profile.empty:
        return None

    sorted_dates = sorted(profile["date"].tolist())

    # Requirement 1: Need at least N days of history
    if len(sorted_dates) <= lookback_days:
        return None
    date_after_lookback = sorted_dates[lookback_days]

    # Requirement 2: Discard first M months
    first_date = sorted_dates[0]
    if isinstance(first_date, datetime):
        first_date_dt = first_date
    else:
        first_date_dt = datetime.combine(first_date, datetime.min.time())

    from dateutil.relativedelta import relativedelta
    date_after_discard = (first_date_dt + relativedelta(months=discard_months)).date()

    # Use the later of the two requirements
    valid_from = max(date_after_lookback, date_after_discard)
    return valid_from


def extract_file_start_date(tick_file: Path) -> Optional[datetime]:
    """Extract the year-month start date from a tick filename.

    Expects format like: BTCUSDT-aggTrades-2020-01.parquet

    Returns:
        datetime for the first day of that month, or None if parsing fails.
    """
    stem = tick_file.stem
    parts = stem.split("-")
    if len(parts) >= 4:
        try:
            year = int(parts[-2])
            month = int(parts[-1])
            return datetime(year, month, 1).date()
        except (ValueError, IndexError):
            pass
    return None
