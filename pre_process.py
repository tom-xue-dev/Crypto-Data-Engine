import multiprocessing
import sys
from datetime import datetime,timezone
from pandas.compat.numpy.function import validate_take
from path_utils import *
import pandas as pd
import numpy as np
import os
import psutil
from numba import njit
from typing import Optional

def convert_timestamp(ts):
    """将13位或16位时间戳转换为UTC时间"""
    ts_str = str(ts)
    if len(ts_str) == 13:  # 13位毫秒级时间戳
        seconds = ts / 1000.0
    elif len(ts_str) == 16:  # 16位微秒级时间戳
        seconds = ts / 1000000.0
    else:
        raise ValueError("不支持的时间戳位数（需13或16位）")

    return datetime.fromtimestamp(seconds, tz=timezone.utc)



@njit
def get_volume_bar_indices(volume_arr, threshold):
    result = []
    cusum = 0
    for i in range(len(volume_arr)):
        cusum += volume_arr[i]
        if cusum >= threshold:
            cusum = 0  # 达到阈值，重置累加器
            result.append(i)
    return result


def build_bar(segment):
    return {
        'start_time': convert_timestamp(segment['timestamp'].iloc[0]),
        'open': segment['price'].iloc[0],
        'high': segment['price'].max(),
        'low': segment['price'].min(),
        'close': segment['price'].iloc[-1],
        'volume': segment['quantity'].sum(),
        'sell_volume': segment[segment['isBuyerMaker']]['quantity'].sum(),
        'buy_volume': segment[~segment['isBuyerMaker']]['quantity'].sum(),
        'vwap': (segment['price']*segment['quantity']).sum() / segment['quantity'].sum(),
    }



class BarConstructor:
    def __init__(self,folder_path:str, threshold = 100000,bar_type = 'dollar_bar'):
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

    def _construct_dollar_bar(self,data):
        # 找到block_index出现跃迁的位置
        dollar_volume = data['price'] * data['quantity']
        crosses = get_volume_bar_indices(dollar_volume.values,self.threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx:end_idx + 1]
            bars.append(build_bar(segment))
            start_idx = end_idx + 1

        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            bars.append(build_bar(segment))
        bars = pd.DataFrame(bars)
        return bars

    def _construct_volume_bar(self,data):
        crosses = get_volume_bar_indices(data['quantity'].values, self.threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx:end_idx + 1]
            bars.append(build_bar(segment))
            start_idx = end_idx + 1

        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            bars.append(build_bar(segment))
        bars = pd.DataFrame(bars)
        return bars
    def _construct_tick_bar(self,data):
        segments = [data.iloc[i:i + self.threshold] for i in range(0, len(data), self.threshold)]
        bars = []
        for seg in segments:
            bars.append(build_bar(seg))
        bars = pd.DataFrame(bars)
        return bars
    def _construct_imblance_volume_bar(self,data):
        crosses = get_volume_bar_indices(data['quantity'].values, self.threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx:end_idx + 1]
    def _construct_imblance_dollar_bar(self,data):
        pass

    def _construct_tick_run_bar(self,data):
        pass

    def process_asset_data(self) -> Optional[pd.DataFrame]:
        """
        遍历文件夹内所有 CSV 文件（aggTrades数据），分块读取并构建 volume bar。
        """
        # 找到所有 CSV 文件
        # cnt = 0
        dataframes = []
        for file in self.folder_path:
            print(f"Processing file: {file}")
            df = pd.read_parquet(file)
            dataframes.append(df)
        # 最后将所有的 DataFrame 合并成一个
        data = pd.concat(dataframes, ignore_index=True)
        data.columns = self.col_names
        if self.bar_type == 'dollar_bar':
            bars_df = self._construct_dollar_bar(data)

        elif self.bar_type == 'tick_bar':
            bars_df = self._construct_tick_bar(data)
        else:
            bars_df = None
        return bars_df

def run_asset_data(path, asset_name, threshold):
    print(f"start running asset:{asset_name}")
    constructor = BarConstructor(folder_path=path, threshold=threshold, bar_type='tick_bar')
    df = constructor.process_asset_data()
    df.index = pd.MultiIndex.from_arrays([df['start_time'], [asset_name] * len(df)], names=['time', 'asset'])
    df = df.drop(columns=['start_time'])
    print(df)
    df.to_parquet(f'./data_aggr/dollar_bar/{asset_name}.parquet')
    return asset_name  # 可选：返回处理完成的 asset 名称


if __name__ == "__main__":

    assets = get_sorted_assets(root_dir=r'.\data',end="USDT")
    results = []
    with multiprocessing.Pool(processes=4) as pool:
        for asset, size in assets:
            process_num = min(16, 16 / (size / 1024 / 1024 / 1024 * 6))
            if process_num < 16:
                break
            if process_num < 4:
                print('assets stop at', assets)
                break
            paths = get_asset_file_path(asset)
            if paths[0].split("-")[2] == '2025':
                continue  # 过滤掉刚上线的资产
            bar_size = 10000
            r = pool.apply_async(run_asset_data, args=(paths, asset, bar_size))
            results.append(r)

        for r in results:
            print(f'{r.get()} done!')



