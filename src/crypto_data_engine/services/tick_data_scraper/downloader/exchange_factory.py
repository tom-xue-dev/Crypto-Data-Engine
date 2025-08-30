from typing import Dict, List
from .binance import BinanceAdapter
from .exchange_adapter import ExchangeAdapter
from .okx import OKXAdapter


class ExchangeFactory:
    """交易所适配器工厂"""
    _adapters = {
        'binance': BinanceAdapter,
        'okx': OKXAdapter,
        # 可以继续添加更多交易所
    }

    @classmethod
    def create_adapter(cls, exchange_name: str, config: Dict) -> ExchangeAdapter:
        """创建交易所适配器实例"""
        exchange_name = exchange_name.lower()
        if exchange_name not in cls._adapters:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        adapter_class = cls._adapters[exchange_name]
        return adapter_class(config)

    @classmethod
    def get_supported_exchanges(cls) -> List[str]:
        """获取支持的交易所列表"""
        return list(cls._adapters.keys())

    @classmethod
    def register_adapter(cls, exchange_name: str, adapter_class):
        """注册新的交易所适配器"""
        cls._adapters[exchange_name.lower()] = adapter_class