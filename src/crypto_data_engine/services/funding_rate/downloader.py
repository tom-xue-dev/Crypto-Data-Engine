"""
Binance Futures funding rate downloader.

Downloads funding rate history via REST API pagination and stores as monthly Parquet files.
Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate
  - limit: max 1000 per request
  - startTime / endTime: millisecond timestamps
  - No authentication required
  - Rate limit: 500 requests per 5 minutes per IP
"""
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from crypto_data_engine.common.config.paths import FUTURES_DATA_ROOT
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

FUNDING_RATE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
FUNDING_RATE_DIR = FUTURES_DATA_ROOT / "funding_rate"
MAX_RECORDS_PER_REQUEST = 1000
MAX_RETRIES = 3

# Global rate limiter: ensure minimum interval between any two requests
_rate_lock = threading.Lock()
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.05  # ~20 req/s, well under 2400 weight/min limit


def _rate_limited_get(url: str, params: dict, timeout: int = 15) -> requests.Response:
    """Thread-safe rate-limited GET request."""
    global _last_request_time
    with _rate_lock:
        elapsed = time.time() - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()
    return requests.get(url, params=params, timeout=timeout)


class FundingRateDownloader:
    """Downloads Binance Futures funding rate data via REST API pagination."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or FUNDING_RATE_DIR

    def get_symbols(self) -> List[str]:
        """Get all TRADING PERPETUAL USDT-M futures symbols."""
        resp = requests.get(EXCHANGE_INFO_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return sorted([
            s["symbol"]
            for s in data.get("symbols", [])
            if s.get("status") == "TRADING"
            and s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
        ])

    def _fetch_page(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Dict]:
        """Fetch one page of funding rate data with retry and rate limiting."""
        params = {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": MAX_RECORDS_PER_REQUEST,
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = _rate_limited_get(FUNDING_RATE_URL, params)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(
                        f"Rate limited for {symbol}, waiting {wait}s"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning(
                    f"Fetch failed {symbol} attempt {attempt}/{MAX_RETRIES}: {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
        return []

    def download_symbol_month(
        self,
        symbol: str,
        year: int,
        month: int,
    ) -> Optional[Path]:
        """Download all funding rates for a symbol in a given month.

        Returns path to saved Parquet, or None if no data or already exists.
        """
        sym_dir = self.output_dir / symbol
        out_path = sym_dir / f"{symbol}-fundingRate-{year}-{month:02d}.parquet"

        if out_path.exists():
            return None

        start_dt = datetime(year, month, 1, tzinfo=timezone.utc)
        end_dt = start_dt + relativedelta(months=1)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) - 1

        all_records: List[Dict] = []
        cursor_ms = start_ms

        while cursor_ms < end_ms:
            page = self._fetch_page(symbol, cursor_ms, end_ms)
            if not page:
                break
            all_records.extend(page)
            last_time = page[-1]["fundingTime"]
            if last_time <= cursor_ms:
                break
            cursor_ms = last_time + 1
            if len(page) < MAX_RECORDS_PER_REQUEST:
                break

        if not all_records:
            return None

        df = pd.DataFrame(all_records)
        df = df.rename(columns={
            "fundingTime": "timestamp",
            "fundingRate": "funding_rate",
            "markPrice": "mark_price",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["funding_rate"])
        df["mark_price"] = pd.to_numeric(df["mark_price"])
        df = df[["timestamp", "funding_rate", "mark_price"]]
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        sym_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {symbol} {year}-{month:02d}: {len(df)} records")
        return out_path

    def _probe_first_funding_time(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[datetime]:
        """Probe the earliest funding rate timestamp for a symbol.

        Makes a single API call with the full date range to find the first record.
        Returns the datetime of the first funding rate, or None if no data.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m").replace(tzinfo=timezone.utc)
        end_dt = end_dt + relativedelta(months=1)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) - 1

        page = self._fetch_page(symbol, start_ms, end_ms)
        if not page:
            return None
        first_ts = page[0]["fundingTime"]
        return datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)

    def download_symbol_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """Download funding rates for a symbol across a date range (YYYY-MM format).

        Probes the first available month to skip empty early months.
        Returns number of new months downloaded.
        """
        end_dt = datetime.strptime(end_date, "%Y-%m")

        # Probe earliest data to skip empty months
        first_time = self._probe_first_funding_time(symbol, start_date, end_date)
        if first_time is None:
            return 0

        # Start from the month of the first available record
        effective_start = max(
            datetime.strptime(start_date, "%Y-%m"),
            datetime(first_time.year, first_time.month, 1),
        )

        current = effective_start
        count = 0
        while current <= end_dt:
            result = self.download_symbol_month(
                symbol, current.year, current.month
            )
            if result:
                count += 1
            current += relativedelta(months=1)
        return count
