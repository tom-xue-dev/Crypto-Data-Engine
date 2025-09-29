from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd




class ExchangeAdapter(ABC):
    """Abstract base class for exchange adapters."""
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['base_url']
        self.name = config['exchange_name']
        self.rate_limit = config.get('rate_limit', 1000)

    @abstractmethod
    def get_all_symbols(self, suffix_filter: Optional[str] = None) -> List[str]:
        """Return all trading pairs."""
        pass

    @abstractmethod
    def build_download_url(self, symbol: str, year: int, month: int) -> str:
        """Construct download URL."""
        pass

    @abstractmethod
    def build_checksum_url(self, symbol: str, year: int, month: int) -> str:
        """Construct checksum URL."""
        pass

    @abstractmethod
    def get_file_name(self, symbol: str, year: int, month: int) -> str:
        """Return file name."""
        pass

    @abstractmethod
    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data format."""
        pass
