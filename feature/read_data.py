import os
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import mmap
import gc
from typing import List, Optional
import mplfinance as mpf



class DataLoader:
    def __init__(self, folder: str = ".././data_aggr/dollar_bar",
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 asset_list: Optional[List[str]] = None,file_format = 'parquet',
                 use_cache: bool = True, use_mmap: bool = True,file_end = None):
        self.end = file_end
        self.folder = folder
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.asset_list = asset_list if asset_list else self._get_all_symbol()
        self.use_cache = use_cache
        self.use_mmap = use_mmap
        self.file_format = file_format
        self.cache = {}
        self._check_params_valid()

    def _check_params_valid(self):
        if not self.asset_list:
            raise Exception("asset_list is empty")
        if not os.path.exists(self.folder) or not os.path.isdir(self.folder) :
            raise Exception(f"path{self.folder} doesn't exist or is not a directory")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise Exception("start_date is later than end_date")

    def _read_parquet_file(self, path: str) -> pd.DataFrame:
        if self.use_mmap:
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    buffer = pa.py_buffer(mm)
                    reader = pa.BufferReader(buffer)
                    table = pq.read_table(reader)
                    df = table.to_pandas()
                    if self.start_date and df.index.get_level_values(0)[0] >= self.start_date:
                        print(f'notice:file:{path}\'s earliest date is later than the start_date, skip it.')
                        return pd.DataFrame()
                finally:
                    del reader, buffer, table
                    mm.close()
        else:
            df = pq.read_table(path).to_pandas()
        return df

    def load_all_data(self) -> pd.DataFrame:
        all_data = []
        for symbol in self.asset_list:
            print(f"loading {symbol}.........")
            path = os.path.join(self.folder,symbol+'.'+self.file_format)
            if self.use_cache and symbol in self.cache:
                df = self.cache[symbol]
            else:
                df = self._read_parquet_file(path)
                if self.use_cache:
                    self.cache[symbol] = df

            if self.start_date or self.end_date:
                df['datetime'] = pd.to_datetime(df['datetime'])
                if self.start_date:
                    df = df[df['datetime'] >= self.start_date]
                if self.end_date:
                    df = df[df['datetime'] <= self.end_date]

            all_data.append(df)

        if all_data:
            return pd.concat(all_data).sort_values(by='time')
        else:
            return pd.DataFrame()

    def _get_all_symbol(self):
        symbol_list = []
        for file in os.listdir(self.folder):
            symbol = os.path.splitext(file)[0]
            if self.end is not None:
                if symbol.endswith(self.end):
                    symbol_list.append(symbol)
            else:
                symbol_list.append(symbol)
        return symbol_list

if __name__ == "__main__":
    data_loader = DataLoader(file_end='USDT')

    data = data_loader.load_all_data()
    for asset,group in data.groupby('asset'):
        print(group.index.get_level_values(0)[0],group.index.get_level_values(1)[0],len(group))
        group = group.xs(group.index.get_level_values(1)[0], level='asset')
        # 2. 确保列名符合 mpf 要求（open → Open 等）
        group = group.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'  # 如果你没有 'volume' 列，可用 buy_volume 代替
        })
        pd.set_option('display.max_columns', None)
        # print(group)
        #mpf.plot(group, type='candle', volume=True, style='charles')
        #print()
    # for key,items in data_loader.cache.items():
    #     print(key,items)
    # print(data_loader.cache)


