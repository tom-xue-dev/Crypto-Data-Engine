"""
Binance USDT-M Futures exchange adapter.

Downloads aggTrades data from Binance Vision data archive for all USDT-M perpetual futures.
Data URL pattern: https://data.binance.vision/data/futures/um/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month}.zip
"""
import requests
import pandas as pd
from typing import List, Optional, Dict
from .exchange_adapter import ExchangeAdapter
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

BINANCE_FUTURES_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_TICKER_24H_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_FUTURES_DATA_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades"


class BinanceFuturesAdapter(ExchangeAdapter):
    """Binance USDT-M Futures exchange adapter.

    Downloads historical aggTrades data for all USDT-M perpetual futures
    from the Binance Vision data archive.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.exchange_info_url = config.get(
            "symbol_info_url", BINANCE_FUTURES_EXCHANGE_INFO_URL
        )
        self.supports_checksum = config.get("supports_checksum", True)

    def get_all_symbols(self, suffix_filter: Optional[str] = None) -> List[str]:
        """Retrieve all TRADING-status USDT-M perpetual futures symbols.

        Args:
            suffix_filter: Optional suffix to filter symbols (e.g. "USDT").

        Returns:
            Sorted list of symbol strings.
        """
        try:
            response = requests.get(self.exchange_info_url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as request_error:
            logger.error(f"Failed to fetch Binance Futures exchange info: {request_error}")
            raise

        symbols = [
            symbol_info["symbol"]
            for symbol_info in data.get("symbols", [])
            if (
                symbol_info.get("status") == "TRADING"
                and symbol_info.get("contractType") == "PERPETUAL"
                and symbol_info.get("quoteAsset") == "USDT"
            )
        ]

        if suffix_filter:
            symbols = [s for s in symbols if s.endswith(suffix_filter)]

        symbols.sort()
        logger.info(f"Found {len(symbols)} USDT-M perpetual futures symbols")
        return symbols

    def get_top_symbols_by_volume(self, top_n: int = 100) -> List[str]:
        """Get top N symbols ranked by 24h quote volume.

        This is used by the AssetPoolSelector service, not by the downloader.

        Args:
            top_n: Number of top symbols to return.

        Returns:
            List of symbol strings sorted by descending 24h volume.
        """
        try:
            response = requests.get(BINANCE_FUTURES_TICKER_24H_URL, timeout=15)
            response.raise_for_status()
            tickers = response.json()
        except requests.RequestException as request_error:
            logger.error(f"Failed to fetch Binance Futures 24h tickers: {request_error}")
            raise

        # Filter only USDT pairs and sort by quote volume descending
        usdt_tickers = [
            ticker for ticker in tickers
            if ticker.get("symbol", "").endswith("USDT")
        ]
        usdt_tickers.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)

        top_symbols = [ticker["symbol"] for ticker in usdt_tickers[:top_n]]
        logger.info(
            f"Top {top_n} USDT-M futures by 24h volume: "
            f"{top_symbols[:5]}... (total {len(top_symbols)})"
        )
        return top_symbols

    def build_download_url(self, symbol: str, year: int, month: int) -> str:
        """Build Binance Futures aggTrades download URL.

        URL pattern: {base_url}/{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip
        """
        file_name = self.get_file_name(symbol, year, month)
        return f"{self.base_url}/{symbol}/{file_name}"

    def build_checksum_url(self, symbol: str, year: int, month: int) -> str:
        """Build checksum URL for Binance Futures.

        Pattern: {download_url}.CHECKSUM
        """
        download_url = self.build_download_url(symbol, year, month)
        return f"{download_url}.CHECKSUM"

    def get_file_name(self, symbol: str, year: int, month: int) -> str:
        """Return Binance Futures aggTrades file name."""
        date_str = f"{year}-{month:02d}"
        return f"{symbol}-aggTrades-{date_str}.zip"

    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw Binance Futures aggTrades data.

        Binance Futures aggTrades CSV format (same as spot):
        [aggTradeId, price, quantity, firstTradeId, lastTradeId, timestamp, isBuyerMaker, isBestMatch]
        """
        if data.empty:
            return data

        data.columns = [
            "agg_trade_id", "price", "quantity", "first_trade_id",
            "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match",
        ]
        data["price"] = pd.to_numeric(data["price"])
        data["quantity"] = pd.to_numeric(data["quantity"])
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        return data
