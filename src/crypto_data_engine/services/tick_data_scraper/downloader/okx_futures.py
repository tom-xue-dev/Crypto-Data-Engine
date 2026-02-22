import requests
import pandas as pd
from typing import List, Optional, Dict
from crypto_data_engine.common.logger.logger import get_logger
from .exchange_adapter import ExchangeAdapter

logger = get_logger(__name__)

OKX_FUTURES_EXCHANGE_INFO_URL = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
OKX_FUTURES_DATA_BASE_URL = "https://www.okx.com/cdn/okex/traderecords/trades/monthly"


class OKXFuturesAdapter(ExchangeAdapter):
    """OKX SWAP Futures exchange adapter.

    Downloads historical trades data for all SWAP perpetual futures
    from the OKX CDN archive.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.exchange_info_url = config.get(
            "symbol_info_url", OKX_FUTURES_EXCHANGE_INFO_URL
        )
        self.base_url = config.get("base_url", OKX_FUTURES_DATA_BASE_URL)
        self.supports_checksum = config.get("supports_checksum", False)
        self._list_time_cache: Dict[str, int] = {}

    def get_all_symbols(self, suffix_filter: Optional[str] = None) -> List[str]:
        """Retrieve all OKX SWAP trading pairs.

        Args:
            suffix_filter: Optional suffix to filter symbols (e.g. "-SWAP").

        Returns:
            Sorted list of symbol strings.
        """
        try:
            response = requests.get(self.exchange_info_url, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != "0":
                raise Exception(f"Failed to get OKX symbols: {data.get('msg')}")
        except requests.RequestException as request_error:
            logger.error(f"Failed to fetch OKX Futures exchange info: {request_error}")
            raise

        symbols = []
        for inst in data.get("data", []):
            sym = inst["instId"]
            symbols.append(sym)
            try:
                self._list_time_cache[sym] = int(inst.get("listTime", 0))
            except (ValueError, TypeError):
                self._list_time_cache[sym] = 0

        if suffix_filter:
            symbols = [s for s in symbols if s.endswith(suffix_filter)]

        symbols.sort()
        logger.info(f"Found {len(symbols)} OKX SWAP futures symbols")
        return symbols

    def get_symbol_list_time(self, symbol: str) -> Optional[int]:
        """Return the listing timestamp (ms)."""
        if not self._list_time_cache:
            self.get_all_symbols()
        return self._list_time_cache.get(symbol)

    def build_download_url(self, symbol: str, year: int, month: int) -> str:
        """Build OKX Futures trades download URL.

        URL pattern: {base_url}/{year}{month:02d}/{symbol}-trades-{year}-{month:02d}.zip
        """
        file_name = self.get_file_name(symbol, year, month)
        date_folder = f"{year}{month:02d}"
        return f"{self.base_url}/{date_folder}/{file_name}"

    def build_checksum_url(self, symbol: str, year: int, month: int) -> str:
        """Build checksum URL for OKX Futures."""
        return ""

    def get_file_name(self, symbol: str, year: int, month: int) -> str:
        """Return OKX Futures trades file name."""
        date_str = f"{year}-{month:02d}"
        return f"{symbol}-trades-{date_str}.zip"

    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw OKX Futures trades data.
        """
        return data
