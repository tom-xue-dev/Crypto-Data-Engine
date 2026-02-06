"""
Asset pool selector for dynamic universe management.

Selects the top N assets by trading volume from Binance USDT-M Futures,
with configurable update frequency and caching (Redis or local JSON).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

BINANCE_FUTURES_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"

# Symbols to always exclude (stablecoins, delisted, etc.)
DEFAULT_EXCLUDE_SYMBOLS: Set[str] = {
    "BUSDUSDT",
    "USDCUSDT",
    "TUSDUSDT",
    "FDUSDUSDT",
    "USTUSDT",
    "DAIUSDT",
}


@dataclass
class AssetPoolConfig:
    """Configuration for the asset pool selector."""
    top_n: int = 100
    exchange: str = "binance_futures"
    update_interval_hours: int = 24 * 30  # ~monthly
    exclude_symbols: Set[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_SYMBOLS.copy())
    cache_dir: Optional[Path] = None
    redis_url: Optional[str] = None
    redis_key: str = "asset_pool:current"


class AssetPoolSelector:
    """
    Dynamic asset pool selector.

    Queries Binance Futures 24h ticker data, ranks by quote volume,
    and returns the top N symbols. Results are cached to local JSON
    or Redis to avoid unnecessary API calls.
    """

    def __init__(self, config: Optional[AssetPoolConfig] = None):
        self.config = config or AssetPoolConfig()
        self._cached_pool: Optional[Dict] = None
        self._last_update: float = 0.0

    def get_current_pool(self, top_n: Optional[int] = None, force_refresh: bool = False) -> List[str]:
        """Get the current asset pool.

        Args:
            top_n: Override default top_n from config.
            force_refresh: Force re-fetch from API even if cache is valid.

        Returns:
            List of symbol strings (e.g. ["BTCUSDT", "ETHUSDT", ...]).
        """
        effective_top_n = top_n or self.config.top_n
        cache_age_hours = (time.time() - self._last_update) / 3600

        if not force_refresh and self._cached_pool and cache_age_hours < self.config.update_interval_hours:
            return self._cached_pool["symbols"][:effective_top_n]

        # Try loading from persistent cache
        if not force_refresh:
            cached = self._load_from_cache()
            if cached:
                cache_time = cached.get("updated_at", 0)
                age_hours = (time.time() - cache_time) / 3600
                if age_hours < self.config.update_interval_hours:
                    self._cached_pool = cached
                    self._last_update = cache_time
                    return cached["symbols"][:effective_top_n]

        # Fetch fresh data from API
        symbols = self._fetch_top_symbols_by_volume(effective_top_n)
        pool_data = {
            "symbols": symbols,
            "top_n": effective_top_n,
            "updated_at": time.time(),
            "updated_at_human": datetime.now(timezone.utc).isoformat(),
            "exchange": self.config.exchange,
        }
        self._cached_pool = pool_data
        self._last_update = time.time()
        self._save_to_cache(pool_data)

        return symbols

    def update_pool(self) -> List[str]:
        """Force update the asset pool (re-fetch from API)."""
        return self.get_current_pool(force_refresh=True)

    def _fetch_top_symbols_by_volume(self, top_n: int) -> List[str]:
        """Fetch and rank symbols by 24h quote volume from Binance Futures API."""
        try:
            response = requests.get(BINANCE_FUTURES_TICKER_URL, timeout=15)
            response.raise_for_status()
            tickers = response.json()
        except requests.RequestException as error:
            logger.error(f"Failed to fetch Binance Futures 24h tickers: {error}")
            # Fall back to cached pool if available
            if self._cached_pool:
                logger.warning("Using stale cached pool as fallback")
                return self._cached_pool["symbols"][:top_n]
            raise

        # Filter USDT pairs and exclude unwanted symbols
        usdt_tickers = [
            ticker for ticker in tickers
            if (
                ticker.get("symbol", "").endswith("USDT")
                and ticker.get("symbol") not in self.config.exclude_symbols
            )
        ]

        # Sort by 24h quote volume descending
        usdt_tickers.sort(
            key=lambda ticker: float(ticker.get("quoteVolume", 0)),
            reverse=True,
        )

        top_symbols = [ticker["symbol"] for ticker in usdt_tickers[:top_n]]
        logger.info(
            f"Asset pool updated: {len(top_symbols)} symbols selected "
            f"(top 5: {top_symbols[:5]})"
        )
        return top_symbols

    def _load_from_cache(self) -> Optional[Dict]:
        """Load pool data from persistent cache (local JSON file)."""
        cache_path = self._get_cache_path()
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as file_handle:
                    return json.load(file_handle)
            except (json.JSONDecodeError, IOError) as error:
                logger.warning(f"Failed to load asset pool cache: {error}")
        return None

    def _save_to_cache(self, pool_data: Dict) -> None:
        """Save pool data to persistent cache (local JSON file)."""
        cache_path = self._get_cache_path()
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as file_handle:
                    json.dump(pool_data, file_handle, indent=2, ensure_ascii=False)
                logger.info(f"Asset pool cached to {cache_path}")
            except IOError as error:
                logger.warning(f"Failed to save asset pool cache: {error}")

    def _get_cache_path(self) -> Optional[Path]:
        """Get the cache file path."""
        if self.config.cache_dir:
            return self.config.cache_dir / "asset_pool.json"
        # Default: use E:\data directory
        from crypto_data_engine.common.config.paths import FUTURES_DATA_ROOT
        return FUTURES_DATA_ROOT / "asset_pool.json"

    def get_pool_info(self) -> Dict:
        """Get metadata about the current pool."""
        if self._cached_pool:
            return {
                "exchange": self._cached_pool.get("exchange"),
                "symbol_count": len(self._cached_pool.get("symbols", [])),
                "updated_at": self._cached_pool.get("updated_at_human"),
                "top_5": self._cached_pool.get("symbols", [])[:5],
            }
        return {"status": "not_initialized"}
