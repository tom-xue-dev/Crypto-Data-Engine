import multiprocessing
import os
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import mmap
import gc
from typing import List, Optional
import mplfinance as mpf
from dataclasses import dataclass
import yaml

@dataclass
class DataLoaderConfig:
    folder: str = ".././tick_data/data_aggr/tick_bar"
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    asset_list: Optional[List[str]] = None
    file_format: str = 'parquet'
    use_cache: bool = True
    use_mmap: bool = True
    parallel: bool = True
    file_end: Optional[str] = "USDT"

    @classmethod
    def load(cls, path: str) -> 'DataLoaderConfig':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} not found.")
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        if 'start_date' in config_dict and config_dict['start_date']:
            config_dict['start_date'] = pd.to_datetime(config_dict['start_date']).tz_localize("UTC")
        if 'end_date' in config_dict and config_dict['end_date']:
            config_dict['end_date'] = pd.to_datetime(config_dict['end_date']).tz_localize("UTC")
        return cls(**config_dict)



class DataLoader:
    def __init__(self, config:DataLoaderConfig):
        self.config = config
        self.folder = config.folder
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.use_cache = config.use_cache
        self.use_mmap = config.use_mmap
        self.file_format = config.file_format
        self.end = config.file_end
        self.cache = {}
        self.parallel = config.parallel
        self.asset_list = config.asset_list if config.asset_list else self._get_all_symbol()
        self._check_params_valid()

    def _check_params_valid(self):
        if not self.asset_list:
            raise Exception("asset_list is empty")
        if not os.path.exists(self.folder) or not os.path.isdir(self.folder) :
            raise Exception(f"path{self.folder} doesn't exist or is not a directory")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise Exception("start_date is later than end_date")

    def _read_parquet_file(self, path: str) -> pd.DataFrame:
        df = None
        if self.use_mmap:
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    buffer = pa.py_buffer(mm)
                    reader = pa.BufferReader(buffer)
                    table = pq.read_table(reader)
                    df = table.to_pandas()
                finally:
                    del reader, buffer, table
                    mm.close()
        else:
            df = pq.read_table(path).to_pandas()
        if self.start_date and df.index.get_level_values(0)[0] >= self.start_date:
            print(f'notice:file:{path}\'s earliest date is later than the start_date, skip it.the earliest date is {df.index.get_level_values(0)[0] }')
            return pd.DataFrame()
        else:
            return df

    def load_all_data(self) -> pd.DataFrame:
        all_data = []
        results = None
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._read_parquet_file, [os.path.join(self.folder, symbol + '.' + self.file_format)
                                                         for symbol in self.asset_list])
        for df in results:
            if df.empty:
                continue
            if self.start_date or self.end_date:
                if self.start_date:
                    time_index = df.index.get_level_values('time')
                    df = df[time_index >= self.start_date]
                if self.end_date:
                    time_index = df.index.get_level_values('time')
                    df = df[time_index <= self.end_date]
            all_data.append(df)
        if all_data:
            print("start merging......")
            return pd.concat(all_data, axis=0, sort=False).sort_values(by='time')

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
   config = DataLoaderConfig.load("load_config.yaml")
   data_loader = DataLoader(config)
   df = data_loader.load_all_data()
   print(df.head())
   print(df.tail())
   print(df.shape)
   print(df.info())
   print(df.describe())
   print(df.index.get_level_values(0)[0])
   print(df.index.get_level_values(0)[-1])

