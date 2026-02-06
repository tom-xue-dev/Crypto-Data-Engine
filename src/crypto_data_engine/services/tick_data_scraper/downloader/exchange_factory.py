from typing import Dict, List
from .binance import BinanceAdapter
from .binance_futures import BinanceFuturesAdapter
from .exchange_adapter import ExchangeAdapter
from .okx import OKXAdapter


class ExchangeFactory:
    """Factory for exchange adapters."""
    _adapters = {
        "binance": BinanceAdapter,
        "binance_futures": BinanceFuturesAdapter,
        "okx": OKXAdapter,
    }

    @classmethod
    def create_adapter(cls, exchange_name: str, config: Dict) -> ExchangeAdapter:
        """Create an adapter instance for the specified exchange."""
        exchange_name = exchange_name.lower()
        if exchange_name not in cls._adapters:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        adapter_class = cls._adapters[exchange_name]
        return adapter_class(config)

    @classmethod
    def get_supported_exchanges(cls) -> List[str]:
        """Return list of supported exchanges."""
        return list(cls._adapters.keys())

    @classmethod
    def register_adapter(cls, exchange_name: str, adapter_class):
        """Register a new exchange adapter."""
        cls._adapters[exchange_name.lower()] = adapter_class