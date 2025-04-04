import os
import pandas as pd
import multiprocessing
from datetime import datetime
from Config import Config
from bar_constructor import BarConstructor
from path_utils import get_sorted_assets, get_asset_file_path
import numpy as np
class PreprocessContext:
    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.aggr_dir = config.get('aggregated_data_dir', './data_aggr')
        self.bar_type = config.get('bar_type', 'volume_bar')
        self.threshold = config.get('threshold', 10000000)
        self.process_num_limit = config.get('process_num_limit', 4)
        self.suffix_filter = config.get('suffix_filter', None)
        self.adaptive = config.get('adaptive', False)
        self.sample_days = config.get('sample_days', 1)
        self.target_bars = config.get('target_bars', 300)


def suggest_threshold_by_sample(bar_type,asset_files, target_bars=300, sample_days=1):
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

    def safe_convert(ts):
        ts_str = str(ts)
        if len(ts_str) == 13:
            return pd.to_datetime(ts, unit='ms', utc=True)
        elif len(ts_str) >= 16:
            return pd.to_datetime(ts / 1000, unit='ms', utc=True)
        return pd.NaT

    all_samples = []
    for file in asset_files:
        df = pd.read_parquet(file)
        df.columns = col_names
        df['time'] = df['timestamp'].apply(safe_convert)
        df['date'] = df['time'].dt.date

        unique_dates = df['date'].drop_duplicates().tolist()
        for date in unique_dates:
            if len(all_samples) >= sample_days:
                break
            sample = df[df['date'] == date]
            all_samples.append(sample)

        if len(all_samples) >= sample_days:
            break

    if not all_samples:
        return 10000000

    sample_df = pd.concat(all_samples, ignore_index=True)
    if bar_type == 'tick_bar':
        tick_count = len(sample_df)
        threshold = max(tick_count // target_bars, 300)
    elif bar_type == 'dollar_bar':
        dollar_count = np.sum(sample_df['price'] * sample_df['quantity'].abs())
        threshold = max(dollar_count.sum() // target_bars, 1000)
    elif bar_type == 'volume_bar':
        volume_count = np.sum(sample_df['quantity'])
        threshold = max(volume_count // target_bars, 1000)
    else:
        raise ValueError(f"Unsupported bar type: {bar_type}")
    return threshold



def run_asset_data(path, asset, context, threshold):
    constructor = BarConstructor(folder_path=path, threshold=threshold, bar_type=context.bar_type)
    df = constructor.process_asset_data()

    df.index = pd.MultiIndex.from_arrays(
        [df['start_time'], [asset] * len(df)], names=['time', 'asset']
    )
    df = df.drop(columns=['start_time'])

    output_dir = os.path.join(context.aggr_dir, context.bar_type)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{asset}.parquet")
    df.to_parquet(output_path)

    print(f"✅ {asset} 已完成 (threshold={threshold})")
    return asset


def process_asset(args):
    asset, size, context = args
    if not asset.endswith('USDT'):
        return None

    paths = get_asset_file_path(asset, data_dir=context.data_dir)
    if not paths or paths[0][-15:-11] == '2025':
        return None

    if context.adaptive:
        threshold = suggest_threshold_by_sample(bar_type=context.bar_type, asset_files=paths,
        target_bars=context.target_bars, sample_days=context.sample_days
        )
        print(f"[INFO] {asset} → Adaptive Threshold = {threshold}")
    else:
        threshold = context.threshold

    return run_asset_data(paths, asset, context, threshold)


def run_pipeline(config):
    context = PreprocessContext(config)
    assets = get_sorted_assets(root_dir=context.data_dir, suffix_filter=context.suffix_filter)

    filtered_assets = []
    for asset, size in assets:
        paths = get_asset_file_path(asset, data_dir=context.data_dir)
        if not paths or paths[0][-15:-11] == '2025':
            continue
        filtered_assets.append((asset, size))

    asset_infos = [(asset, size, context) for asset, size in filtered_assets]

    with multiprocessing.Pool(processes=context.process_num_limit) as pool:
        for res in pool.imap_unordered(process_asset, asset_infos):
            if res:
                print(f"{res} done!")

if __name__ == "__main__":
    config = Config(path="preprocessor.yaml")
    run_pipeline(config)