import os
import pandas as pd
import numpy as np
from numba import njit
from datetime import datetime, timezone

@njit
def get_volume_bar_indices(volume_arr, threshold):
    indices = []
    cum_volume = 0
    for i in range(len(volume_arr)):
        cum_volume += volume_arr[i]
        if cum_volume >= threshold:
            indices.append(i)
            cum_volume = 0
    return indices

class BarConstructor:
    def __init__(self, folder_path, threshold=100000, bar_type="dollar_bar"):
        self.bar_type = bar_type
        self.folder_path = folder_path
        self.threshold = threshold
        self.col_names = [
            "aggTradeId",
            "price",
            "quantity",
            "firstTradeId",
            "lastTradeId",
            "timestamp",
            "isBuyerMaker",
            "isBestMatch"
        ]

    def _construct_dollar_bar(self, data):
        dollar_volume = data['price'] * data['quantity']
        crosses = get_volume_bar_indices(dollar_volume.values, self.threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx:end_idx + 1]
            bars.append(self.build_bar(segment, bar_type='dollar_bar'))
            start_idx = end_idx + 1

        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            bars.append(self.build_bar(segment, bar_type='dollar_bar'))
        bars = pd.DataFrame(bars)
        bars.columns = [str(col) for col in bars.columns]
        return bars

    def _construct_volume_bar(self, data):
        crosses = get_volume_bar_indices(data['quantity'].values, self.threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx:end_idx + 1]
            bars.append(self.build_bar(segment, bar_type='volume_bar'))
            start_idx = end_idx + 1

        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            bars.append(self.build_bar(segment, bar_type='volume_bar'))
        bars = pd.DataFrame(bars)
        bars.columns = [str(col) for col in bars.columns]
        return bars

    def _construct_tick_bar(self, data):
        segments = [data.iloc[i:i + self.threshold] for i in range(0, len(data), self.threshold)]
        bars = []
        for seg in segments:
            bars.append(self.build_bar(seg, bar_type='tick_bar'))
        bars = pd.DataFrame(bars)
        bars.columns = [str(col) for col in bars.columns]
        return bars

    def build_bar(self, segment, bar_type='tick_bar'):
        if bar_type == 'tick_bar':
            return {
                'start_time': self.convert_timestamp(segment['timestamp'].iloc[0]),
                'open': segment['price'].iloc[0],
                'high': segment['price'].max(),
                'low': segment['price'].min(),
                'close': segment['price'].iloc[-1],
                'volume': segment['quantity'].sum(),
                'sell_volume': segment[segment['isBuyerMaker']]['quantity'].sum(),
                'buy_volume': segment[~segment['isBuyerMaker']]['quantity'].sum(),
                'vwap': (segment['price'] * segment['quantity']).sum() / segment['quantity'].sum(),
                'medium_price': segment['price'].median(),
                'volume_std': segment['quantity'].std(),
                'tick_interval_mean': segment['timestamp'].diff().mean(),
                'best_match': segment['isBestMatch'].sum() / len(segment),
            }
        elif bar_type in ['dollar_bar', 'volume_bar']:
            return {
                'start_time': self.convert_timestamp(segment['timestamp'].iloc[0]),
                'open': segment['price'].iloc[0],
                'high': segment['price'].max(),
                'low': segment['price'].min(),
                'close': segment['price'].iloc[-1],
                'volume': segment['quantity'].sum(),
                'sell_volume': segment[segment['isBuyerMaker']]['quantity'].sum(),
                'buy_volume': segment[~segment['isBuyerMaker']]['quantity'].sum(),
                'vwap': (segment['price'] * segment['quantity']).sum() / segment['quantity'].sum(),
                'best_match_ratio': segment['isBestMatch'].sum() / len(segment),
            }

    def convert_timestamp(self, ts):
        length = len(str(ts))
        if length == 13:  # 毫秒
            ts = ts / 1000
        elif length >= 16:  # 微秒
            ts = ts / 1000000
        else:
            raise ValueError(f"Unsupported timestamp length: {length}")
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def process_asset_data(self):
        dataframes = []
        for file in self.folder_path:
            df = pd.read_parquet(file)
            dataframes.append(df)
        data = pd.concat(dataframes, ignore_index=True)
        data.columns = self.col_names

        if self.bar_type == 'dollar_bar':
            bars_df = self._construct_dollar_bar(data)
        elif self.bar_type == 'tick_bar':
            bars_df = self._construct_tick_bar(data)
        elif self.bar_type == 'volume_bar':
            bars_df = self._construct_volume_bar(data)
        else:
            bars_df = None

        return bars_df
