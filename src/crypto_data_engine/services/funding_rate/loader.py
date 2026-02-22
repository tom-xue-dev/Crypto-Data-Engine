"""
Funding rate data loader and daily aggregator for backtest integration.

Usage:
    from crypto_data_engine.services.funding_rate.loader import load_daily_funding_rates

    # Returns {symbol: {date: daily_funding_rate}}
    fr_daily = load_daily_funding_rates(["BTCUSDT", "ETHUSDT"])

    # In backtest PnL loop:
    # pnl -= weight * fr_daily.get(sym, {}).get(d, 0.0)
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from crypto_data_engine.common.config.paths import FUTURES_DATA_ROOT
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

FUNDING_RATE_DIR = FUTURES_DATA_ROOT / "funding_rate"


def load_funding_rates(
    symbol: str,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load all funding rate parquet files for a symbol.

    Returns DataFrame with columns: timestamp, funding_rate, mark_price
    """
    root = data_dir or FUNDING_RATE_DIR
    sym_dir = root / symbol
    empty = pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"])

    if not sym_dir.exists():
        return empty

    files = sorted(sym_dir.glob("*.parquet"))
    if not files:
        return empty

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return empty

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df


def load_daily_funding_rates(
    symbols: List[str],
    data_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Load funding rates for multiple symbols, aggregated to daily.

    Funding settles 3x daily (00:00, 08:00, 16:00 UTC).
    Daily funding rate = sum of the intraday settlement rates.

    Returns:
        {symbol: {datetime.date: daily_funding_rate}}
    """
    result: Dict[str, Dict] = {}
    loaded = 0
    for sym in symbols:
        df = load_funding_rates(sym, data_dir)
        if df.empty:
            continue
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date")["funding_rate"].sum()
        result[sym] = daily.to_dict()
        loaded += 1

    logger.info(f"Loaded daily funding rates for {loaded}/{len(symbols)} symbols")
    return result
