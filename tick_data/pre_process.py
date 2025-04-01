import os
import multiprocessing
import re

from tqdm import tqdm
import pandas as pd
from Config import Config
from bar_constructor import BarConstructor
from path_utils import get_sorted_assets, get_asset_file_path

class PreprocessContextTag:
    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.aggr_dir = config.get('aggregated_data_dir', './data_aggr_tag')
        self.bar_type = config.get('bar_type', 'tick_bar')
        self.threshold = config.get('threshold', 1000)
        self.process_num_limit = config.get('process_num_limit', 4)
        self.suffix_filter = config.get('suffix_filter', None)


def run_asset_data(path, asset, context):
    constructor = BarConstructor(folder_path=path, threshold=context.threshold, bar_type=context.bar_type)
    df = constructor.process_asset_data()

    df.index = pd.MultiIndex.from_arrays(
        [df['start_time'], [asset] * len(df)], names=['time', 'asset']
    )
    df = df.drop(columns=['start_time'])

    output_dir = os.path.join(context.aggr_dir, context.bar_type)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{asset}.parquet")
    df.to_parquet(output_path, compression="brotli")

    print(f"✅ {asset} 已完成")
    return asset

def run_pipeline(config):
    context = PreprocessContextTag(config)

    assets = get_sorted_assets(root_dir=context.data_dir, suffix_filter=context.suffix_filter)

    with multiprocessing.Pool(processes=context.process_num_limit) as pool:
        results = []
        for asset, size in tqdm(assets, desc="[Preprocessing Assets - Tag Mode]"):
            paths = get_asset_file_path(asset, data_dir=context.data_dir)
            if paths[0][-15:-9] == '2025':
                continue  # 过滤掉刚上线的资产
            r = pool.apply_async(run_asset_data, args=(paths, asset, context))
            results.append(r)

        for r in results:
            print(f"{r.get()} done!")


if __name__ == "__main__":
    config = Config(path="preprocessor.yaml")
    run_pipeline(config)