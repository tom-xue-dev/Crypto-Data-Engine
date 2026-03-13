"""
Binance USDT-M Futures bookTicker exchange adapter.

Downloads bookTicker data from Binance Vision data archive.
Data URL pattern:
    https://data.binance.vision/data/futures/um/monthly/bookTicker/{symbol}/{symbol}-bookTicker-{year}-{month}.zip

BookTicker CSV columns:
    update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
    transaction_time, event_time
"""
import requests
import pandas as pd
from typing import List, Optional, Dict

from .exchange_adapter import ExchangeAdapter
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

BINANCE_FUTURES_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_BOOKTICKER_BASE_URL = (
    "https://data.binance.vision/data/futures/um/monthly/bookTicker"
)


class BinanceFuturesBookTickerAdapter(ExchangeAdapter):
    """Binance USDT-M Futures bookTicker adapter.

    Downloads historical best bid/ask (bookTicker) data for USDT-M
    perpetual futures from the Binance Vision data archive.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.exchange_info_url = config.get(
            "symbol_info_url", BINANCE_FUTURES_EXCHANGE_INFO_URL
        )
        self.supports_checksum = config.get("supports_checksum", True)
        self._list_time_cache: Dict[str, int] = {}
        # Use bookTicker-specific base URL (may be overridden by data_type config)
        self.base_url = config.get("base_url", BINANCE_FUTURES_BOOKTICKER_BASE_URL)

    def get_all_symbols(self, suffix_filter: Optional[str] = None) -> List[str]:
        """Retrieve all TRADING-status USDT-M perpetual futures symbols."""
        try:
            response = requests.get(self.exchange_info_url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as request_error:
            logger.error(f"Failed to fetch Binance Futures exchange info: {request_error}")
            raise

        symbols = []
        for symbol_info in data.get("symbols", []):
            if (
                symbol_info.get("status") == "TRADING"
                and symbol_info.get("contractType") == "PERPETUAL"
                and symbol_info.get("quoteAsset") == "USDT"
            ):
                sym = symbol_info["symbol"]
                symbols.append(sym)
                self._list_time_cache[sym] = symbol_info.get("onboardDate")

        if suffix_filter:
            symbols = [s for s in symbols if s.endswith(suffix_filter)]

        symbols.sort()
        logger.info(f"Found {len(symbols)} USDT-M perpetual futures symbols (bookTicker)")
        return symbols

    def get_symbol_list_time(self, symbol: str) -> Optional[int]:
        """Return the listing timestamp (ms)."""
        if not self._list_time_cache:
            self.get_all_symbols()
        return self._list_time_cache.get(symbol)

    def build_download_url(self, symbol: str, year: int, month: int) -> str:
        """Build Binance Futures bookTicker download URL.

        URL pattern: {base_url}/{symbol}/{symbol}-bookTicker-{year}-{month:02d}.zip
        """
        file_name = self.get_file_name(symbol, year, month)
        return f"{self.base_url}/{symbol}/{file_name}"

    def build_checksum_url(self, symbol: str, year: int, month: int) -> str:
        """Build checksum URL for bookTicker data."""
        download_url = self.build_download_url(symbol, year, month)
        return f"{download_url}.CHECKSUM"

    def get_file_name(self, symbol: str, year: int, month: int) -> str:
        """Return Binance Futures bookTicker file name."""
        date_str = f"{year}-{month:02d}"
        return f"{symbol}-bookTicker-{date_str}.zip"

    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw Binance Futures bookTicker data.

        BookTicker CSV columns:
        [update_id, best_bid_price, best_bid_qty, best_ask_price,
         best_ask_qty, transaction_time, event_time]
        """
        if data.empty:
            return data

        data.columns = [
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "transaction_time",
            "event_time",
        ]
        data["best_bid_price"] = pd.to_numeric(data["best_bid_price"])
        data["best_bid_qty"] = pd.to_numeric(data["best_bid_qty"])
        data["best_ask_price"] = pd.to_numeric(data["best_ask_price"])
        data["best_ask_qty"] = pd.to_numeric(data["best_ask_qty"])
        data["transaction_time"] = pd.to_datetime(
            data["transaction_time"], unit="ms"
        )
        data["event_time"] = pd.to_datetime(data["event_time"], unit="ms")
        return data
