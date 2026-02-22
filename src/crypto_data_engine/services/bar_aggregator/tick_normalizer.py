"""
Tick data normalization layer.

Provides a single entry point to convert raw exchange tick DataFrames
(with varying column names and timestamp units) into a standard schema
expected by all downstream aggregators.

Standard output schema:
    timestamp      int64    (milliseconds since epoch)
    price          float64
    quantity       float64
    is_buyer_maker bool

Optional columns (preserved if present):
    agg_trade_id, first_trade_id, last_trade_id
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Standard column names expected by all aggregators
# =============================================================================
STANDARD_COLUMNS = ["timestamp", "price", "quantity", "is_buyer_maker"]
REQUIRED_COLUMNS = ["timestamp", "price", "quantity"]

# =============================================================================
# Column name aliases → standard name
# =============================================================================
COLUMN_ALIAS_MAP: Dict[str, str] = {
    # Timestamp variants
    "transact_time": "timestamp",
    "trade_time": "timestamp",
    "time": "timestamp",
    "ts": "timestamp",
    "T": "timestamp",
    # Price variants
    "p": "price",
    "px": "price",
    # Quantity variants
    "qty": "quantity",
    "sz": "quantity",
    "q": "quantity",
    "size": "quantity",
    "amount": "quantity",
    # Buyer/maker variants
    "isBuyerMaker": "is_buyer_maker",
    "is_buyer": "is_buyer_maker",
    "buyer_maker": "is_buyer_maker",
    "m": "is_buyer_maker",
    # OKX specific variants
    "created_time": "timestamp",
}

# Binance aggTrades CSV has no header; when converted to parquet the columns
# become integer indices 0–7. Map them to standard names.
NUMERIC_INDEX_MAP: Dict[int, str] = {
    0: "agg_trade_id",
    1: "price",
    2: "quantity",
    3: "first_trade_id",
    4: "last_trade_id",
    5: "timestamp",
    6: "is_buyer_maker",
}


# =============================================================================
# Public API
# =============================================================================

def normalize_tick_data(
    dataframe: pd.DataFrame,
    source_hint: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a raw tick DataFrame to the standard aggregator schema.

    Handles:
    - Numeric column indices (legacy headerless CSV → parquet)
    - Alias column names (transact_time, isBuyerMaker, qty, etc.)
    - Timestamp unit detection and conversion to milliseconds
    - Data type enforcement (price/quantity → float64, is_buyer_maker → bool)

    Args:
        dataframe: Raw tick data with arbitrary column naming.
        source_hint: Optional filename for better error messages.

    Returns:
        DataFrame with standard columns and millisecond timestamps.

    Raises:
        ValueError: If required columns (timestamp, price, quantity) cannot
                    be resolved from the input.
    """
    result = dataframe.copy()

    # Step 1: Resolve column names
    result = _resolve_column_names(result)

    # Step 2: Validate required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in result.columns]
    if missing:
        hint = f" (source: {source_hint})" if source_hint else ""
        raise ValueError(
            f"Missing required columns after normalization: {missing}. "
            f"Available: {list(result.columns)}{hint}"
        )

    # Step 3: Normalize timestamp to milliseconds
    result["timestamp"] = _normalize_timestamps_to_ms(result["timestamp"])

    # Step 4: Enforce data types
    result["price"] = result["price"].astype(np.float64)
    result["quantity"] = result["quantity"].astype(np.float64)

    if "is_buyer_maker" in result.columns:
        result["is_buyer_maker"] = result["is_buyer_maker"].astype(bool)
    elif "side" in result.columns:
        result["is_buyer_maker"] = result["side"].str.lower().eq("sell")
        result = result.drop(columns=["side"], errors="ignore")
    else:
        result["is_buyer_maker"] = False

    return result


# =============================================================================
# Internal helpers
# =============================================================================

def _resolve_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Map raw column names (numeric indices or aliases) to standard names."""
    if len(dataframe.columns) == 0:
        return dataframe

    needs_positional_mapping = _has_positional_columns(dataframe)

    if needs_positional_mapping:
        # Reset columns to integer indices and apply positional mapping
        dataframe.columns = range(len(dataframe.columns))
        rename_map = {
            idx: name
            for idx, name in NUMERIC_INDEX_MAP.items()
            if idx < len(dataframe.columns)
        }
        dataframe = dataframe.rename(columns=rename_map)
        # Drop leftover numeric columns (e.g. column 7 in Binance aggTrades)
        dataframe = dataframe[[
            col for col in dataframe.columns if isinstance(col, str)
        ]]
        return dataframe

    # Handle string alias column names
    rename_map = {
        old: new
        for old, new in COLUMN_ALIAS_MAP.items()
        if old in dataframe.columns and new not in dataframe.columns
    }
    if rename_map:
        dataframe = dataframe.rename(columns=rename_map)

    return dataframe


def _has_positional_columns(dataframe: pd.DataFrame) -> bool:
    """Detect whether column names are positional (not semantic).

    Returns True for:
    - Numeric columns: [0, 1, 2, ...] (int or numpy.int64)
    - Data-as-header: ['35488350', '0.008824', '11196', ...] (first row used as header)

    Returns False for named columns like ['price', 'quantity', 'transact_time', ...].
    """
    first_col = dataframe.columns[0]

    # Numeric type (int, numpy.int64, etc.)
    if not isinstance(first_col, str):
        return True

    # Known semantic column names — definitely not positional
    known_names = set(COLUMN_ALIAS_MAP.keys()) | set(STANDARD_COLUMNS) | {
        "agg_trade_id", "first_trade_id", "last_trade_id",
    }
    if first_col in known_names:
        return False

    # Heuristic: if none of the required columns exist AND the first column
    # looks like a numeric value, this is likely data-as-header
    has_any_required = any(col in dataframe.columns for col in REQUIRED_COLUMNS)
    has_any_alias = any(col in dataframe.columns for col in COLUMN_ALIAS_MAP)
    if has_any_required or has_any_alias:
        return False

    # Final check: try to parse the first column name as a number
    try:
        float(first_col)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_timestamps_to_ms(series: pd.Series) -> pd.Series:
    """Detect timestamp unit from digit count and convert to milliseconds.

    Supports:
    - 10 digits → seconds    (multiply by 1000)
    - 13 digits → milliseconds (no change)
    - 16+ digits → microseconds (integer divide by 1000)
    """
    if series.empty:
        return series

    sample_value = int(series.iloc[0])
    digit_count = len(str(abs(sample_value)))

    if digit_count >= 16:
        # Microseconds → milliseconds
        return series // 1000
    elif digit_count >= 13:
        # Already milliseconds
        return series
    elif digit_count >= 10:
        # Seconds → milliseconds
        return series * 1000
    else:
        logger.warning(
            f"Unrecognized timestamp unit ({digit_count} digits, "
            f"sample={sample_value}). Assuming milliseconds."
        )
        return series
