import requests
from .exchange_adapter import ExchangeAdapter
import pandas as pd
from typing import List, Optional, Dict


class BinanceAdapter(ExchangeAdapter):
    """Binance交易所适配器"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"

    def get_all_symbols(self, suffix_filter: Optional[str] = None) -> List[str]:
        """获取所有Binance交易对"""
        response = requests.get(self.exchange_info_url)
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']

        if suffix_filter:
            symbols = [s for s in symbols if s.endswith(suffix_filter)]
        return symbols

    def build_download_url(self, symbol: str, year: int, month: int) -> str:
        """构建Binance下载URL"""
        file_name = self.get_file_name(symbol, year, month)
        return f"{self.base_url}/{symbol}/{file_name}"

    def build_checksum_url(self, symbol: str, year: int, month: int) -> str:
        """构建Binance校验和URL"""
        file_name = self.get_file_name(symbol, year, month)
        return f"{self.base_url}/{symbol}/{file_name}.CHECKSUM"

    def get_file_name(self, symbol: str, year: int, month: int) -> str:
        """获取Binance文件名格式"""
        date_str = f"{year}-{month:02d}"
        return f"{symbol}-aggTrades-{date_str}.zip"

    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理Binance原始数据格式"""
        # Binance aggTrades格式: [aggTradeId, price, quantity, firstTradeId, lastTradeId, timestamp, isBuyerMaker, isBestMatch]
        if data.empty:
            return data

        data.columns = [
            'agg_trade_id', 'price', 'quantity', 'first_trade_id',
            'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match'
        ]

        # 转换数据类型
        data['price'] = pd.to_numeric(data['price'])
        data['quantity'] = pd.to_numeric(data['quantity'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        return data

    def validate_symbol(self, symbol: str) -> bool:
        """验证Binance交易对格式"""
        return symbol.isalpha() and len(symbol) >= 6