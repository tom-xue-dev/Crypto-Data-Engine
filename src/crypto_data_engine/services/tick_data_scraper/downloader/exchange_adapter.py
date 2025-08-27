from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd




class ExchangeAdapter(ABC):
    """交易所适配器抽象基类"""
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['base_url']
        self.name = config['exchange_name']
        self.rate_limit = config.get('rate_limit', 1000)

    @abstractmethod
    def get_all_symbols(self, suffix_filter: Optional[str] = None) -> List[str]:
        """获取所有交易对"""
        pass

    @abstractmethod
    def build_download_url(self, symbol: str, year: int, month: int) -> str:
        """构建下载URL"""
        pass

    @abstractmethod
    def build_checksum_url(self, symbol: str, year: int, month: int) -> str:
        """构建校验和URL"""
        pass

    @abstractmethod
    def get_file_name(self, symbol: str, year: int, month: int) -> str:
        """获取文件名"""
        pass

    @abstractmethod
    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理原始数据格式"""
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """验证交易对是否有效"""
        pass