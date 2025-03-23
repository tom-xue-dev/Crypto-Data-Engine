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


def save_parquet(data, path):
    data:pd.DataFrame
    data.to_parquet(path)


class BarConstructor:
    def __init__(self, asset_name, volume_threshold=10000):
        self.asset_name = asset_name
        self.volume_threshold = volume_threshold

    def reset_accumulator(self):
        """
        初始化累计器，保存当前 bar 的累积数据
        ohlcv:开盘收盘最高最低和成交量
        start_time:bar开始的时间
        buy_volume:主动买入成交量
        sell_volume:主动卖出成交量
        dollar_volume:成交额
        num_trades:一共多少笔交易
        total_price:所有交易价格总和
        """
        return {
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0,
            'start_time': None,
            'buy_volume': 0,
            'sell_volume': 0,
            'dollar_volume': 0,
            'num_trades': 0,
            'total_price': 0,

        }


    def update_accumulator(self,acc, row):
        """
        更新累计器。
        row 中至少要包含 'price' 和 'quantity' 这两个字段。
        """
        price = row['price']
        qty = row['quantity']
        ts = convert_timestamp(row['timestamp'])  # 这里是已经转成 datetime 的时间戳

        if acc['open'] is None:
            # 第一次赋值
            acc['open'] = price
            acc['high'] = price
            acc['low'] = price
            acc['close'] = price
            acc['volume'] += qty
            acc['start_time'] = ts
        else:
            acc['close'] = price
            acc['high'] = max(acc['high'], price)
            acc['low'] = min(acc['low'], price)
            acc['volume'] += qty
        acc['buy_volume'] += acc['volume'] if not row['isBuyerMaker'] else 0
        acc['sell_volume'] += acc['volume'] if row['isBuyerMaker'] else 0
        acc['dollar_volume'] += acc['volume'] * price
        acc['num_trades'] += 1
        acc['total_price'] += price
        return acc


    def extract_bars_from_data(self,data, acc, volume_threshold=10000):
        """
        遍历 chunk 中每一行数据，流式构建 volume bar。
        """
        quantities = data['quantity'].values
        offset = acc['volume'] if acc['volume'] is not None else 0.0
        cum_vol = np.cumsum(quantities) + offset
        # 如果整个文件都没达到一个完整 bar，则直接更新 acc 并返回空列表
        if cum_vol[-1] < volume_threshold:
            # 更新累计器：逐行累加（因为数据量较小）
            for i in range(len(data)):
                acc = self.update_accumulator(acc, data.iloc[i])
            return [], acc

        thresholds = np.arange(volume_threshold, cum_vol[-1] + volume_threshold, volume_threshold)
        thresholds = thresholds[thresholds <= cum_vol[-1]]
        indices = np.searchsorted(cum_vol, thresholds)
        indices = np.unique(indices)
        bars = []
        start_idx = 0
        # 对于第一个 bar，如果 acc 中已有累计数据，则使用其 open 和 start_time
        for idx in indices:
            # 取出从 start_idx 到 idx（包含 idx 行）的数据作为一个完整 bar
            bar_slice = data.iloc[start_idx: idx + 1]
            # 聚合 open、high、low、close、volume、start_time 等指标
            bar = {}
            if acc['open'] is not None:
                bar['start_time'] = acc['start_time']
                bar['open'] = acc['open']
            else:
                ts = convert_timestamp(bar_slice.iloc[0]['timestamp'])
                bar['start_time'] = ts
                bar['open'] = bar_slice.iloc[0]['price']

            bar['high'] = max(acc['high'] if acc['high'] is not None else -np.inf, bar_slice['price'].max())
            bar['low'] = min(acc['low'] if acc['low'] is not None else np.inf, bar_slice['price'].min())
            bar['close'] = bar_slice.iloc[-1]['price']
            # 计算 bar 的成交量时，累加前文件残留的 volume 和本文件的切片数据
            bar['volume'] = acc['volume'] + bar_slice['quantity'].sum()
            bar['buy_volume'] = bar_slice[bar_slice['isBuyerMaker']]['quantity'].sum() + acc['buy_volume']
            bar['sell_volume'] = bar_slice[~bar_slice['isBuyerMaker']]['quantity'].sum() + acc['sell_volume']
            bar['total_price'] = acc['total_price'] + bar_slice['price'].sum()
            bar['num_trades'] = acc['num_trades'] + len(bar_slice)
            bar['dollar_volume'] = bar['total_price'] * bar['volume']
            # 如果需要，还可以计算 dollar_volume、num_trades 等

            bars.append(bar)
            # 重置累积器，因为 bar 已经完整
            acc = self.reset_accumulator()
            # 更新起始索引为下一个 bar 的开始
            start_idx = idx + 1

        # 对剩余部分（未完成一个完整 bar）更新累积器
        if start_idx < len(data):
            for i in range(start_idx, len(data)):
                acc = self.update_accumulator(acc, data.iloc[i])
        return bars, acc

    def construct_dollar_bar(self,data, threshold):
        # 找到block_index出现跃迁的位置
        dollar_volume = data['price'] * data['quantity']
        crosses = get_volume_bar_indices(dollar_volume.values,threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx:end_idx + 1]
            bar = {
                'start_time': convert_timestamp(segment['timestamp'].iloc[0]),
                'open': segment['price'].iloc[0],
                'high': segment['price'].max(),
                'low': segment['price'].min(),
                'close': segment['price'].iloc[-1],
                'volume': segment['quantity'].sum(),
                'sell_volume': segment[segment['isBuyerMaker']]['quantity'].sum(),
                'buy_volume': segment[~segment['isBuyerMaker']]['quantity'].sum()
            }
            bars.append(bar)
            start_idx = end_idx + 1
        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            bar = {
                'start_time': convert_timestamp(segment['timestamp'].iloc[0]),
                'open': segment['price'].iloc[0],
                'high': segment['price'].max(),
                'low': segment['price'].min(),
                'close': segment['price'].iloc[-1],
                'volume': segment['quantity'].sum(),
                'sell_volume': segment[segment['isBuyerMaker']]['quantity'].sum(),
                'buy_volume': segment[~segment['isBuyerMaker']]['quantity'].sum()
            }
            bars.append(bar)
        bars = pd.DataFrame(bars)
        return bars

    def process_asset_data(self,assets_filepath, threshold=150000):
        """
        遍历文件夹内所有 CSV 文件（aggTrades数据），分块读取并构建 volume bar。
        """
        col_names = [
            "aggTradeId",
            "price",
            "quantity",
            "firstTradeId",
            "lastTradeId",
            "timestamp",
            "isBuyerMaker",
            "isBestMatch"
        ]
        # 找到所有 CSV 文件
        all_bars = []
        acc = self.reset_accumulator()  # 全局累计器，用于跨文件连续累加
        # cnt = 0
        dataframes = []
        for file in assets_filepath:
            print(f"Processing file: {file}")
            df = pd.read_parquet(file)
            dataframes.append(df)
        # 最后将所有的 DataFrame 合并成一个
        data = pd.concat(dataframes, ignore_index=True)
        data.columns = col_names
        bars_df = self.construct_dollar_bar(data,threshold)
        # all_bars.extend(bars)
        # bars_df = pd.DataFrame(all_bars)
        return bars_df

def run_asset_data(paths, asset, bar_size):
    print(f"start running asset:{asset}")
    constructor = BarConstructor(asset)
    df = constructor.process_asset_data(paths, bar_size)
    df.index = pd.MultiIndex.from_arrays([df['start_time'], [asset] * len(df)], names=['time', 'asset'])
    df.to_parquet(f'./data_aggr/dollar_bar/{asset}.parquet')
    return asset  # 可选：返回处理完成的 asset 名称


if __name__ == "__main__":
    assets = get_sorted_assets(root_dir=r'.\data')
    dollar_thresholds = [5000,200000]
    results = []
    with multiprocessing.Pool(processes=8) as pool:
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
            constructor = BarConstructor(asset)
            bar_size = 5000
            r = pool.apply_async(run_asset_data, args=(paths, asset, bar_size))
            results.append(r)

        for r in results:
            print(f'{r.get()} done!')



