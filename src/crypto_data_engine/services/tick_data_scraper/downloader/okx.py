from .exchange_adapter import ExchangeAdapter
import requests
import pandas as pd
from typing import List, Optional, Dict


class OKXAdapter(ExchangeAdapter):
    """OKX交易所适配器"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.exchange_info_url = "https://www.okx.com/api/v5/public/instruments"

    def get_all_symbols(self, suffix_filter: Optional[str] = None) -> List[str]:
        """获取所有OKX交易对"""
        params = {"instType": "SPOT"}
        response = requests.get(self.exchange_info_url, params=params)
        data = response.json()

        if data['code'] != '0':
            raise Exception(f"Failed to get OKX symbols: {data['msg']}")

        symbols = [inst['instId'].replace('-', '') for inst in data['data']]

        if suffix_filter:
            symbols = [s for s in symbols if s.endswith(suffix_filter)]
        return symbols

    def build_download_url(self, symbol: str, year: int, month: int) -> str:
        """构建OKX下载URL"""
        # OKX可能使用不同的URL格式
        file_name = self.get_file_name(symbol, year, month)
        return f"{self.base_url}/spot/{symbol}/{file_name}"

    def build_checksum_url(self, symbol: str, year: int, month: int) -> str:
        """构建OKX校验和URL"""
        file_name = self.get_file_name(symbol, year, month)
        return f"{self.base_url}/spot/{symbol}/{file_name}.sha256"

    def get_file_name(self, symbol: str, year: int, month: int) -> str:
        """获取OKX文件名格式"""
        date_str = f"{year}-{month:02d}"
        return f"{symbol}-trades-{date_str}.zip"

    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理OKX原始数据格式"""
        if data.empty:
            return data

        # OKX trades格式可能不同，需要根据实际格式调整
        data.columns = [
            'trade_id', 'price', 'size', 'side', 'timestamp'
        ]

        data['price'] = pd.to_numeric(data['price'])
        data['size'] = pd.to_numeric(data['size'])
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        return data

    def validate_symbol(self, symbol: str) -> bool:
        """验证OKX交易对格式"""
        return symbol.isalnum() and len(symbol) >= 6